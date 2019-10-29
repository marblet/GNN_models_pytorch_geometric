from functools import partial
from hyperopt import fmin, tpe, hp, STATUS_OK
from math import log
from torch.optim import Adam

from train import run
from models.sgc import create_sgc_model


def objective(dataset, model, lr, space):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=space['weight_decay'])
    evals = run(dataset, model, optimizer, early_stopping=False)
    return {
        'loss': -evals['val_acc'],
        'status': STATUS_OK
        }


def search_best_hp(dataset, model, lr):
    f = partial(objective, dataset, model, lr)
    space = {'weight_decay': hp.loguniform('weight_decay', log(1e-9), log(1e-3))}
    best = fmin(fn=f, space=space, algo=tpe.suggest, max_evals=60)
    print(best['weight_decay'])
    return best['weight_decay']
