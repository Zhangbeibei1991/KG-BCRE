from dgl.nn.pytorch import GraphConv
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # hidden layers
        for i in range(n_layers):
            self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, feat, efeat):
        h = feat
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
        return h