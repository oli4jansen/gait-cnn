import coloredlogs
import torch
from torch.utils.data import DataLoader
import argparse
import logging
from time import time

from dataset import GaitDataset
from models import GaitNet

parser = argparse.ArgumentParser(description='GaitNet')
parser.add_argument('--dataset', type=str, default='data/0-full-clips')


def train(model, dataset):
    # TODO: use dataset_train.classes and dataset_train.class_counts for weighted sampling in data loader
    batch_size = 24
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logging.info(str(len(dataset)) + ' clips in train dataset')
    logging.info(str(len(dataloader)) + ' batches in train dataloader')

    weight = torch.Tensor(dataset.dataset.class_counts)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    NUM_EPOCHS = 15
    for epoch in range(NUM_EPOCHS):
        logging.info(f'epoch: {epoch + 1}/{NUM_EPOCHS}')

        for i, (inputs, labels) in enumerate(dataloader):
            logging.info(f'batch {i} of {len(dataloader)}')

            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward, calculate loss, backward and optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            logging.info(f'loss is {loss}')

    logging.info('\ntraining finished')

def test(model, dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=24)

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            output = model(inputs)
            test_loss += torch.nn.functional.cross_entropy(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))

def main(args):
    init()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = GaitDataset(args.dataset)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    model = GaitNet(num_classes=len(dataset.classes))
    model.to(device)

    train(model=model, dataset=train_set)
    test(model=model, dataset=test_set)

    checkpoint_name = f'checkpoint_{args.dataset}_{int(time())}.pt'
    torch.save(model.state_dict(), checkpoint_name)

    logging.info('checkpoint saved')

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
