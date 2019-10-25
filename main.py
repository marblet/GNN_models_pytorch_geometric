from models.gat import create_gat_model
from models.gcn import create_gcn_model
from train import run

if __name__ == '__main__':
    # GCN
    # dataset, model, optimizer = create_gcn_model('Cora')

    # GAT
    dataset, model, optimizer = create_gat_model('Cora')
    run(dataset, model, optimizer, patience=100)