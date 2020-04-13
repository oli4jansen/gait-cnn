import itertools
import json
import math
import os

import coloredlogs
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
import logging
from statistics import mean

from dataset import GaitDataset
from models import GaitNet

parser = argparse.ArgumentParser(description='GaitNet')
parser.add_argument('--dataset', type=str, default='data/full/preprocessed')
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--lr', type=int, default=0.0001)
parser.add_argument('--epochs', type=int, default=5)


def train(model, dataset, epochs, lr=0.0001):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=24, shuffle=True)

    logging.info(str(len(dataset)) + ' clips in train dataset')
    logging.info(str(len(dataloader)) + ' batches in train dataloader')

    weight = torch.Tensor(dataset.dataset.class_counts)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7, weight_decay=1e-7)

    losses = dict()

    for epoch in range(epochs):
        losses[f'epoch-{epoch}'] = dict()

        for i, (inputs, labels) in enumerate(dataloader):
            logging.info(f'epoch {epoch + 1}/{epochs}, batch {i + 1}/{len(dataloader)}')

            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward, calculate loss, backward and optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            logging.info(f'loss is {loss}')
            losses[f'epoch-{epoch}'][i] = loss.item()

    logging.info('training finished')

    return losses

def test(model, dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=24)

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            test_loss += torch.nn.functional.cross_entropy(outputs, labels).item()
            preds = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += preds.eq(labels.view_as(preds)).sum().item()

    test_loss /= len(dataloader)

    logging.info(f'test loss: {test_loss}')
    logging.info(f'test accuracy: {100. * correct / len(dataloader.dataset)}')

    return test_loss, 100. * correct / len(dataloader.dataset)


def main(args):
    init()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = GaitDataset(args.dataset)
    folds = np.array_split(range(0, len(dataset)), args.k)
    logging.info(f'dataset has {len(dataset)} items')
    logging.info(f'folds will be of length {[len(fold) for fold in folds]}')

    fold_accuracies = []

    for fold in range(0, args.k):
        logging.info(f'starting with fold {fold + 1}')
        model = GaitNet(num_classes=len(dataset.classes))
        model.to(device)

        train_folds = [f for i, f in enumerate(folds) if i != fold]
        train_set = torch.utils.data.Subset(dataset, indices=list(itertools.chain.from_iterable(train_folds)))
        test_set = torch.utils.data.Subset(dataset, indices=folds[fold])

        train_losses = train(model=model, dataset=train_set, epochs=args.epochs)
        test_loss, test_accuracy = test(model=model, dataset=test_set)

        with open(f'train_fold_{fold + 1}.json', 'w+') as file:
            json.dump(train_losses, file)

        with open(f'test_fold_{fold + 1}.json', 'w+') as file:
            json.dump(dict({ 'test_loss': test_loss, 'test_accuracy': test_accuracy }), file)

        checkpoint_name = f'checkpoint_{os.path.basename(args.dataset)}_fold{fold + 1}.pt'
        torch.save(model.state_dict(), checkpoint_name)
        logging.info(f'fold {fold + 1} checkpoint saved')

        fold_accuracies.append(test_accuracy)

    logging.info(f'{args.k}-fold mean accuracy: {mean(fold_accuracies)}')

def init():
    coloredlogs.install(level='INFO', fmt='> %(asctime)s %(levelname)-8s %(message)s')

    # prevents CPU overload that crashes laptop
    if not torch.cuda.is_available():
        logging.info('using CPU, limiting threads')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    else:
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        logging.info('using %s', device_name)
        # Force CUDA tensors by default
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
