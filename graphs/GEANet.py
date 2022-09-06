import torch.nn as nn
import torch.nn.functional as F
import torch

from dgl.nn.pytorch.conv import NNConv


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


class GEANet(NNConv):
    def __init__(self, in_feats, out_feats, edge_func, aggregator_type, args):
        super(GEANet, self).__init__(in_feats, out_feats, edge_func, aggregator_type)
        self.args = args
        self.edge_nn = edge_func  # EdgeEmbedding
        self.att = torch.nn.Linear(2 * out_feats, 1)
        self.in_channels = in_feats
        self.negative_slope = 0.2
        self.reset_parameters()

    def send_attention(self, edge_feat):
        feat_src = edge_feat.src["h"]
        feat_dst = edge_feat.dst["h"]
        weight = edge_feat.data['w']
        neighbors = torch.matmul(feat_src.unsqueeze(1), weight).squeeze(dim=1)
        new_feat = torch.cat([feat_src, neighbors], dim=1)
        alpha = F.softmax(self.att(new_feat).squeeze(), dim=0)
        feat_dst = feat_dst * alpha.view(-1, 1)
        return {"ft": feat_dst}

    def forward(self, graph, feat, efeat):
        with graph.local_scope():
            efeat = graph.edata["h"]
            feat_dst = graph.dstdata["h"]
            graph.edata['w'] = self.edge_nn(efeat).view(-1, self._in_src_feats, self._out_feats)
            graph.apply_edges(self.send_attention)
            def message_func(edge):
                return {'m': edge.data['ft']}

            graph.update_all(message_func=message_func, reduce_func=self.reducer('m', 'ft'))
            # output = graph.ndata.pop('ft')
            rst = graph.dstdata.pop('ft')  # (n, d_out)
            # residual connection
            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)
            # bias
            if self.bias is not None:
                rst = rst + self.bias
            return rst
