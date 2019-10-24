import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, dataset, nhid, heads, dropout):
        super(GAT, self).__init__()
        self.gc1 = GATConv(dataset.num_features, nhid, heads=heads, dropout=dropout)
        self.gc2 = GATConv(nhid, dataset.num_classes, heads=1, dropout=dropout)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)
