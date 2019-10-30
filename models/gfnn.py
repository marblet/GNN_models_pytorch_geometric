import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import SGConv

from datasets import get_planetoid_dataset


class GFNN(nn.Module):
    def __init__(self, dataset, nhid, K):
        super(GFNN, self).__init__()
        self.gc1 = SGConv(dataset.num_features, nhid, K=K, cached=True)
        self.fc1 = nn.Linear(nhid, dataset.num_classes)

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.fc1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


def create_gfnn_model(data_name, nhid=32, lr=0.2, weight_decay=5e-4):
    dataset = get_planetoid_dataset(data_name, True)
    model = GFNN(dataset, nhid, 2)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return dataset, model, optimizer
