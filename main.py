import itertools
import json
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
parser.add_argument('--mode', choices=['kfold', 'full', 'both'], default='both')
parser.add_argument('--dataset', type=str, default='data/full/preprocessed')
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--bs', type=int, default=24)


def train_epoch(criterion, epochs, epoch, model, dataloader, lr):
    losses = dict()

    optimizer = torch.optim.Adam(model.parameters(), eps=1e-7, weight_decay=1e-7, lr=lr)

    for i, (inputs, labels) in enumerate(dataloader):
        logging.info(f'epoch {epoch + 1}/{epochs}, batch {i + 1}/{len(dataloader)}')

        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward, calculate loss, backward and optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Keep track of losses for later debugging
        logging.info(f'loss is {loss}')
        losses[i] = loss.item()

    return losses


def test(model, dataset, batch_size):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    logging.info('starting test')

    model.eval()
    test_losses = []
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            test_losses.append(torch.nn.functional.cross_entropy(outputs, labels).item())
            preds = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += preds.eq(labels.view_as(preds)).sum().item()

    logging.info(f'avg test loss: {mean(test_losses)}')
    logging.info(f'min test loss: {min(test_losses)}')
    logging.info(f'max test loss: {max(test_losses)}')
    logging.info(f'test accuracy: {100. * correct / len(dataloader.dataset)}')

    return test_losses, 100. * correct / len(dataloader.dataset)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = GaitDataset(args.dataset)

    logging.info(f'dataset has {len(dataset)} items')
    logging.info(f'learning rate is {args.lr}')
    logging.info(f'batch size is {args.bs}')

    if args.mode == 'kfold' or args.mode == 'both':
        # Split dataset into folds for k-fold cross validation
        folds = np.array_split(range(0, len(dataset)), args.k)
        logging.info(f'folds will be of size {[len(fold) for fold in folds]}')

        fold_accuracies = []

        for fold in range(0, args.k):
            logging.info(f'starting with fold {fold + 1}/{args.k}')

            # Init new model
            model = GaitNet(num_classes=len(dataset.classes))
            model.to(device)

            # The current fold index will be the test set, others will be train set
            train_folds = [f for i, f in enumerate(folds) if i != fold]
            train_set = torch.utils.data.Subset(dataset, indices=list(itertools.chain.from_iterable(train_folds)))
            test_set = torch.utils.data.Subset(dataset, indices=folds[fold])

            # Train model and save losses to JSON file
            dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True)

            logging.info(str(len(dataset)) + ' clips in train dataset')
            logging.info(str(len(dataloader)) + ' batches in train dataloader')

            # Initialise cross entropy with weights as dataset is not balanced perfectly
            weight = torch.Tensor(train_set.dataset.class_counts)
            criterion = torch.nn.CrossEntropyLoss(weight=weight)

            losses = dict()
            for epoch in range(args.epochs):
                losses[f'epoch-{epoch}-train'] = train_epoch(criterion, epochs=args.epochs, epoch=epoch, model=model,
                                                             dataloader=dataloader, lr=args.lr)

            with open(f'train_fold_{fold + 1}.json', 'w+') as file:
                json.dump(losses, file)

            # Save checkpoint for this fold
            checkpoint_name = f'checkpoint_{os.path.basename(args.dataset)}_fold{fold + 1}.pt'
            torch.save(model.state_dict(), checkpoint_name)
            logging.info(f'fold {fold + 1} checkpoint saved')

            # Test model and save loss and accuracy to JSON file
            test_losses, test_accuracy = test(model=model, dataset=test_set, batch_size=args.bs)
            with open(f'test_fold_{fold + 1}.json', 'w+') as file:
                json.dump(dict({'test_losses': test_losses, 'test_accuracy': test_accuracy}), file)

            fold_accuracies.append(test_accuracy)

        # Report mean accuracy of all folds
        logging.info(f'{args.k}-fold mean accuracy: {mean(fold_accuracies)}')

    if args.mode == 'full' or args.mode == 'both':
        # Init new model
        model = GaitNet(num_classes=len(dataset.classes))
        model.to(device)

        # Train model and save losses to JSON file
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True)

        # Initialise cross entropy with weights as dataset is not balanced perfectly
        weight = torch.Tensor(dataset.class_counts)
        criterion = torch.nn.CrossEntropyLoss(weight=weight)

        losses = dict()
        for epoch in range(args.epochs):
            losses[f'epoch-{epoch}-train'] = train_epoch(criterion, epochs=args.epochs, epoch=epoch, model=model,
                                                         dataloader=dataloader, lr=args.lr)
            # logging.info(f'test accuracy now at {accuracy}')

        with open(f'train_full_model.json', 'w+') as file:
            json.dump(losses, file)

        # Save checkpoint for this fold
        checkpoint_name = f'checkpoint_{os.path.basename(args.dataset)}_full_model.pt'
        torch.save(model.state_dict(), checkpoint_name)
        logging.info('full model checkpoint saved')


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
    init()
    args = parser.parse_args()
    main(args)
