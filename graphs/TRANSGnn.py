import torch.nn as nn
import torch
from dgl.nn.pytorch.softmax import edge_softmax
import torch.nn.functional as F
from dgl import function as fn


class TransformerConv(nn.Module):
    """Implementation of TransformerConv from UniMP
    This is an implementation of the paper Unified Message Passing Model for Semi-Supervised Classification
    (https://arxiv.org/abs/2009.03509).
    Args:

        input_size: The size of the inputs.

        hidden_size: The hidden size for gat.

        activation: (default None) The activation for the output.

        num_heads: (default 4) The head number in transformerconv.

        feat_drop: (default 0.6) Dropout rate for feature.

        attn_drop: (default 0.6) Dropout rate for attention.

        concat: (default True) Whether to concat output heads or average them.
        skip_feat: (default True) Whether to add a skip conect from input to output.
        gate: (default False) Whether to use a gate function in skip conect.
        layer_norm: (default True) Whether to aply layer norm in output
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_heads=4,
                 feat_drop=0.6,
                 attn_drop=0.6,
                 concat=True,
                 skip_feat=True,
                 gate=False,
                 layer_norm=True,
                 activation=None):
        super(TransformerConv, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.concat = concat

        self.q = nn.Linear(input_size, num_heads * hidden_size)
        self.k = nn.Linear(input_size, num_heads * hidden_size)
        self.v = nn.Linear(input_size, num_heads * hidden_size)

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)

        if skip_feat:
            if concat:
                self.skip_feat = nn.Linear(input_size, num_heads * hidden_size)
            else:
                self.skip_feat = nn.Linear(input_size, hidden_size)
        else:
            self.skip_feat = None

        if gate:
            if concat:
                self.gate = nn.Linear(3 * num_heads * hidden_size, 1)
            else:
                self.gate = nn.Linear(3 * hidden_size, 1)
        else:
            self.gate = None

        if layer_norm:
            if self.concat:
                self.layer_norm = nn.LayerNorm(num_heads * hidden_size)
            else:
                self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = None

        if isinstance(activation, str):
            activation = getattr(F, activation)
        self.activation = activation

    def send_attention(self, edge_feat):
        if "edge_feat" in edge_feat.data:
            alpha = edge_feat.dst["q"] * (edge_feat.src["k"] + edge_feat.data['edge_feat'])
            edge_feat.src["v"] = edge_feat.src["v"] + edge_feat.data["edge_feat"]
        else:
            alpha = edge_feat.dst["q"] * edge_feat.src["k"]
        # alpha = edge_feat.dst["q"] * edge_feat.src["k"]
        alpha = torch.sum(alpha, dim=-1)
        return {"alpha": alpha, "v": edge_feat.src["v"]}

    def send_recv(self, graph, q, k, v, edge_feat):
        q = q / (self.hidden_size ** 0.5)
        graph.srcdata['k'] = k
        graph.srcdata['v'] = v
        graph.dstdata['q'] = q
        graph.edata['edge_feat'] = edge_feat

        graph.apply_edges(self.send_attention)
        graph.edata['alpha'] = edge_softmax(graph, graph.edata.pop('alpha'))

        alpha = torch.reshape(graph.edata['alpha'], [-1, self.num_heads, 1])
        if self.attn_drop > 1e-15:
            alpha = self.attn_dropout(alpha)

        feature = graph.edata['v']
        feature = feature * alpha
        if self.concat:
            feature = torch.reshape(feature,
                                    [-1, self.num_heads * self.hidden_size])
        else:
            feature = torch.mean(feature, dim=1)
        graph.edata.pop('alpha')
        graph.edata['ft'] = feature

        def message_func(edge):
            return {'m': edge.data['ft']}

        graph.update_all(message_func=message_func, reduce_func=fn.sum('m', 'ft'))
        output = graph.ndata.pop('ft')
        return output

    def forward(self, graph, feature, edge_feat=None):
        if self.feat_drop > 1e-5:
            feature = self.feat_dropout(feature)
        q = self.q(feature)
        k = self.k(feature)
        v = self.v(feature)

        q = torch.reshape(q, [-1, self.num_heads, self.hidden_size])
        k = torch.reshape(k, [-1, self.num_heads, self.hidden_size])
        v = torch.reshape(v, [-1, self.num_heads, self.hidden_size])

        if edge_feat is not None:
            if self.feat_drop > 1e-5:
                edge_feat = self.feat_dropout(edge_feat)
            edge_feat = torch.reshape(edge_feat,
                                      [-1, self.num_heads, self.hidden_size])

        output = self.send_recv(graph, q, k, v, edge_feat=edge_feat)

        if self.skip_feat is not None:
            skip_feat = self.skip_feat(feature)
            if self.gate is not None:
                gate = F.sigmoid(
                    self.gate(
                        torch.cat(
                            [skip_feat, output, skip_feat - output], dim=-1)))
                output = gate * skip_feat + (1 - gate) * output
            else:
                output = skip_feat + output

        if self.layer_norm is not None:
            output = self.layer_norm(output)

        if self.activation is not None:
            output = self.activation(output)
        return output


class EdgeEmbedding(nn.Module):
    '''
    A neural network ℎ𝚯 that maps edge features edge_attr of shape [-1, num_edge_features] to shape [-1, in_channels * out_channels], e.g., defined by torch.nn.Sequential.
    '''

    def __init__(self, edge_embedding_dim, output_dim, args):
        super(EdgeEmbedding, self).__init__()
        self.args = args
        if args.using_edge:
            self.edge_embedding = nn.Embedding.from_pretrained(args.rel_embeddings, freeze=args.kg_embedding_frozen)
        else:
            self.edge_embedding = nn.Embedding(num_embeddings=args.num_edge_embeddings,
                                               embedding_dim=edge_embedding_dim)
        self.linear2 = nn.Linear(edge_embedding_dim, output_dim)

    def forward(self, edge_attr):
        # sum pooling of embedding
        embedding = torch.matmul(edge_attr.float(), self.edge_embedding.weight)
        x = F.relu(self.linear2(embedding))
        return x

class Transformer(nn.Module):
    """Implement of TransformerConv
    """

    def __init__(self,
                 args,
                 input_size,
                 num_layers=1,
                 drop_rate=0.5,
                 feat_drop=0.6,
                 attn_drop=0.6,
                 num_heads=8,
                 hidden_size=8,
                 activation=None,
                 concat=True,
                 skip_feat=True,
                 gate=False,
                 layer_norm=True):
        super(Transformer, self).__init__()
        self.num_layers = num_layers
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.trans = nn.ModuleList()

        self.EdgeEmbedding = EdgeEmbedding(input_size, input_size, args)

        for i in range(self.num_layers):
            if i == 0:
                self.trans.append(
                    TransformerConv(
                        input_size,
                        self.hidden_size,
                        self.num_heads,
                        self.feat_drop,
                        self.attn_drop,
                        skip_feat=skip_feat,
                        activation=activation,
                        concat=concat,
                        gate=gate,
                        layer_norm=layer_norm,
                    ))
            else:
                self.trans.append(
                    TransformerConv(
                        self.num_heads * self.hidden_size,
                        self.hidden_size,
                        self.num_heads,
                        self.feat_drop,
                        self.attn_drop,
                        skip_feat=skip_feat,
                        activation=activation,
                        concat=concat,
                        gate=gate,
                        layer_norm=layer_norm,
                    ))

            self.trans.append(nn.Dropout(self.drop_rate))

    def forward(self, graph, feat, efeat=None):
        efeat = graph.edata["h"]
        efeat = self.EdgeEmbedding(efeat)
        for m in self.trans:
            if isinstance(m, nn.Dropout):
                feat = m(feat)
            else:
                feat = m(graph, feat, efeat)
        return feat
