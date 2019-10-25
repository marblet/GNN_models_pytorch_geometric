import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import degree

from datasets import get_planetoid_dataset


class MaskedGCN(nn.Module):
    def __init__(self, dataset, nhid, dropout):
        super(MaskedGCN, self).__init__()
        self.gc1 = GCNConv(dataset.num_features, nhid)
        self.gc2 = GCNConv(nhid, dataset.num_classes)
        self.sigma1 = torch.nn.Parameter(torch.rand(dataset.num_features))
        self.sigma2 = torch.nn.Parameter(torch.rand(nhid))
        self.sigma1.requires_grad = True
        self.sigma2.requires_grad = True
        self.dropout = dropout

    def mask_features(self, x, edge_index, sigma):
        deg = degree(edge_index)
        return x

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.mask_features(x, edge_index, self.sigma1)
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mask_features(x, edge_index, self.sigma2)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)


def create_masked_gcn_model(data_name, nhid=16, dropout=0.5,
                     lr=0.01, weight_decay=5e-4):
    dataset = get_planetoid_dataset(data_name, True)
    model = MaskedGCN(dataset, nhid, dropout)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return dataset, model, optimizer