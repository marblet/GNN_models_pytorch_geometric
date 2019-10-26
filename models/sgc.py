import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import SGConv

from datasets import get_planetoid_dataset


class SGC(nn.Module):
    def __init__(self, dataset, K):
        super(SGC, self).__init__()
        self.gc1 = SGConv(dataset.num_features, dataset.num_classes, K=K, cached=True)

    def reset_parameters(self):
        self.gc1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gc1(x, edge_index)
        return F.log_softmax(x, dim=1)


def create_sgc_model(data_name, lr=0.2, weight_decay=3e-5):
    dataset = get_planetoid_dataset(data_name, True)
    model = SGC(dataset, 2)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return dataset, model, optimizer
