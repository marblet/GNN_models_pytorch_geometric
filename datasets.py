from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def get_planetoid_dataset(data_name, normalize_features):
    dataset = Planetoid(root='/tmp/' + data_name, name=data_name)
    if normalize_features:
        dataset.transform = T.NormalizeFeatures()

    return dataset
