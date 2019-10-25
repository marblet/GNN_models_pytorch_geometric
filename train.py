import torch
import torch.nn.functional as F


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output[data.train_mask == 1], data.y[data.train_mask == 1])
    loss.backward()
    optimizer.step()


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        output = model(data)

    outputs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(output[mask == 1], data.y[mask == 1]).item()
        pred = output[mask == 1].max(dim=1)[1]
        acc = pred.eq(data.y[mask == 1]).sum().item() / mask.sum().item()

        outputs['{}_loss'.format(key)] = loss
        outputs['{}_acc'.format(key)] = acc

    return outputs


def run(dataset, model, optimizer, epochs=200, early_stopping=True, patience=10, verbose=False):
    data = dataset[0]

    # for early stopping
    best_val_loss = float('inf')
    counter = 0
    for epoch in range(1, epochs+1):
        train(model, optimizer, data)
        evals = evaluate(model, data)

        if verbose:
            print('epoch:', epoch, 'train loss:', evals['train_loss'],
                  'val loss:', evals['val_loss'])

        if early_stopping:
            if evals['val_loss'] < best_val_loss:
                best_val_loss = evals['val_loss']
                counter = 0
            else:
                counter += 1
            if counter >= patience:
                print("Stop training")
                break
    for met, val in evals.items():
        print(met, val)
