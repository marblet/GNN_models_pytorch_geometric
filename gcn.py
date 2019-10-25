import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv

from datasets import get_planetoid_dataset


class GCN(nn.Module):
    def __init__(self, dataset, nhid, dropout):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(dataset.num_features, nhid)
        self.gc2 = GCNConv(nhid, dataset.num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)


def create_gcn_model(data_name, nhid=16, dropout=0.5,
                     lr=0.01, weight_decay=5e-4):
    dataset = get_planetoid_dataset(data_name, True)
    model = GCN(dataset, nhid, dropout)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return dataset, model, optimizer
