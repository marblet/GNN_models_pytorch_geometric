import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GATConv

from datasets import get_planetoid_dataset


class GAT(nn.Module):
    def __init__(self, dataset, nhid, first_heads, output_heads, dropout):
        super(GAT, self).__init__()
        self.gc1 = GATConv(dataset.num_features, nhid,
                           heads=first_heads, dropout=dropout)
        self.gc2 = GATConv(nhid*first_heads, dataset.num_classes,
                           heads=output_heads, dropout=dropout)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)


def create_gat_model(data_name, nhid=8, first_heads=8, output_heads=1,
                     dropout=0.6, lr=0.005, weight_decay=5e-4):
    dataset = get_planetoid_dataset(data_name, True)
    model = GAT(dataset, nhid, first_heads, output_heads, dropout)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return dataset, model, optimizer
