from dgl import DGLGraph
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl

from dgl.nn.pytorch import RelGraphConv


class RGCN(nn.Module):
    def __init__(self,
                 num_nodes,
                 h_dim,
                 out_dim,
                 num_layers,
                 num_rels,
                 regularizer="basis",
                 num_bases=-1,
                 dropout=0.,
                 self_loop=False,
                 ns_mode=False):
        super(RGCN, self).__init__()

        if num_bases == -1:
            num_bases = num_rels
        self.emb = nn.Embedding(num_nodes, h_dim)

        self.conv = nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = RelGraphConv(h_dim, h_dim, num_rels, regularizer,
                                num_bases, self_loop=self_loop)
            self.conv.append(conv)
        self.conv.append(RelGraphConv(h_dim, out_dim, num_rels, regularizer, num_bases, self_loop=self_loop))
        self.dropout = nn.Dropout(dropout)
        self.ns_mode = ns_mode

    def forward(self, graph, feat, efeat):
        if self.ns_mode:
            # forward for neighbor sampling
            h = feat
            for i in range(len(self.conv) - 1):
                h = self.conv1(graph[0], h, graph[0].edata[dgl.ETYPE], graph[0].edata['norm'])
                h = self.dropout(F.relu(h))
            h = self.conv[-1](graph[1], h, graph[1].edata[dgl.ETYPE], graph[1].edata['norm'])
            return h
        else:
            h = feat
            for i in range(len(self.conv) - 1):
                h = self.conv1(graph, h, graph.edata[dgl.ETYPE], graph.edata['norm'])
                h = self.dropout(F.relu(h))
            h = self.conv[-1](graph, h, graph.edata[dgl.ETYPE], graph.edata['norm'])
            return h
