from models.gcn import create_gcn_model
from models.gat import create_gat_model
from models.masked_gcn import create_masked_gcn_model
from models.sgc import create_sgc_model
from train import run

if __name__ == '__main__':
    # GCN
    # dataset, model, optimizer = create_gcn_model('Cora')
    # run(dataset, model, optimizer, verbose=True)

    # GAT
    # dataset, model, optimizer = create_gat_model('Cora')
    # run(dataset, model, optimizer, patience=100)

    # SGC
    dataset, model, optimizer = create_sgc_model('Cora')
    run(dataset, model, optimizer, epochs=100, early_stopping=False, verbose=False)

    # Masked GCN
    # dataset, model, optimizer = create_masked_gcn_model('Cora')
    # run(dataset, model, optimizer, verbose=True)