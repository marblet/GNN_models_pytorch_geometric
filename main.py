from models.gcn import create_gcn_model
from models.gat import create_gat_model
from models.gfnn import create_gfnn_model
from models.masked_gcn import create_masked_gcn_model
from models.sgc import create_sgc_model
from train import run
from tuning import search_best_hp

if __name__ == '__main__':
    # GCN
    # dataset, model, optimizer = create_gcn_model('Cora')
    # run(dataset, model, optimizer, verbose=True)

    # GAT
    # dataset, model, optimizer = create_gat_model('Cora')
    # run(dataset, model, optimizer, patience=100)

    # SGC
    # Hyper parameter search
    dataset, model, _ = create_sgc_model('Cora')
    weight_decay = search_best_hp(dataset, model, lr=0.2)
    dataset, model, optimizer = create_sgc_model('Cora', weight_decay=weight_decay)
    run(dataset, model, optimizer, epochs=100, early_stopping=False, verbose=False)

    # gfNN
    # Hyper parameter search
    # dataset, model, _ = create_gfnn_model('Cora')
    # weight_decay = search_best_hp(dataset, model, lr=0.2)
    # dataset, model, optimizer = create_gfnn_model('Cora')
    # run(dataset, model, optimizer, epochs=50, early_stopping=False)

    # Masked GCN
    # dataset, model, optimizer = create_masked_gcn_model('Cora')
    # run(dataset, model, optimizer, verbose=True)
