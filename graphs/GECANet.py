import torch.nn as nn
import torch
import torch.nn.functional as F
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
        # self.dropout = nn.Dropout(0.2)
        self.edge_att = nn.Linear(edge_embedding_dim, 1)

    def forward(self, edge_attr):
        if self.args.using_att:
            edge_len = edge_attr.size(0)
            edge_embed = self.edge_embedding.weight.unsqueeze(dim=0).repeat(edge_len, 1, 1)
            edge_weights = self.edge_att(edge_embed).squeeze(dim=-1)
            edge_mask = (1.0 - edge_attr) * -1e10
            edge_weights = torch.softmax(edge_weights * edge_mask, dim=-1).unsqueeze(dim=-1)
            embedding = F.relu(self.linear2(torch.sum(edge_embed * edge_weights,dim=1)))
            return embedding
        else:
            # sum pooling of embedding
            embedding = torch.matmul(edge_attr.float(), self.edge_embedding.weight)
            x = F.relu(self.linear2(embedding))
            return x


class GECANet(NNConv):
    def __init__(self, in_feats, out_feats, edge_func, aggregator_type, args):
        super(GECANet, self).__init__(in_feats, out_feats, edge_func, aggregator_type)
        self.args = args
        self.edge_nn = edge_func  # EdgeEmbedding
        self.att = torch.nn.Linear(3 * out_feats, 1)
        self.in_channels = in_feats
        self.corporation = torch.nn.GRU(input_size=out_feats, hidden_size=out_feats)
        self.negative_slope = 0.2
        self.reset_parameters()

    def send_attention(self, edge_feat):
        feat_src = edge_feat.src["h"]
        feat_dst = edge_feat.dst["h"]
        weight = edge_feat.data['w']
        neighbors = torch.matmul(feat_src.unsqueeze(1), weight).squeeze(dim=1)
        direct_feat = self.corporation(torch.stack([feat_src, neighbors], dim=0))[1][0]
        new_feat = torch.cat([feat_src, neighbors, direct_feat], dim=1)  # 在这里做文章,GRU有向特征
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