import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.optim import Adam
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import add_self_loops, degree

from datasets import get_planetoid_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mask_features(x, edge_index, edge_weight, sigma):
    source, target = edge_index
    h_s, h_t = x[source], x[target]
    h = (h_t - h_s) / sigma
    h = edge_weight.view(-1, 1) * h * h
    mask = torch.zeros(x.size(), device=device)
    mask.index_add_(0, source, h)
    deg = degree(edge_index[0])
    mask = torch.exp(- mask / deg.view(-1, 1))
    x = x * mask
    return x


class MaskedGCNConv(GCNConv):
    def __init__(self, in_channels, out_channels):
        super(MaskedGCNConv, self).__init__(in_channels, out_channels)
        self.sigma = Parameter(torch.Tensor(1, in_channels))
        nn.init.xavier_uniform_(self.sigma.data, gain=1.414)

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result
        x = mask_features(x, edge_index, norm, self.sigma)
        ret = super().forward(x, edge_index, edge_weight)
        return ret


class MaskedGCN(nn.Module):
    def __init__(self, dataset, nhid, dropout):
        super(MaskedGCN, self).__init__()
        self.gc1 = MaskedGCNConv(dataset.num_features, nhid)
        self.gc2 = MaskedGCNConv(nhid, dataset.num_classes)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)


def create_masked_gcn_model(data_name, nhid=16, dropout=0.5,
                     lr=0.02, weight_decay=5e-4):
    dataset = get_planetoid_dataset(data_name, True)
    model = MaskedGCN(dataset, nhid, dropout)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return dataset, model, optimizer