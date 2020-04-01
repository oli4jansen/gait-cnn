import coloredlogs
import torch
from torch.utils.data import DataLoader
import argparse
import logging

from dataset import GaitDataset
from models import GaitNet

parser = argparse.ArgumentParser(description='GaitNet')
parser.add_argument('--dataset', type=str, default='data/synth-cmu-clips')


def train(model, dataset):
    dataset = GaitDataset(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=24, shuffle=True)
    # TODO: use dataset_train.classes and dataset_train.class_counts for weighted sampling in data loader

    logging.info(str(len(dataset)) + ' clips in train dataset')
    logging.info(str(len(dataloader)) + ' batches in train dataloader')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(5):
        logging.info(f'epoch: {epoch}')

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

    torch.save(model.state_dict(), 'checkpoint.pt')

    logging.info('checkpoint saved')

def evaluate(model, dataset):
    model.eval()

    # with torch.no_grad():
    #     for video, target in test_data_loader:
    #         print('-- Start inference --')
    #         print(video.size())
    #         output = model(video)
    #         print(output.size())
    #         print(output)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GaitNet()
    model.to(device)

    train(model=model, dataset=args.dataset)

    evaluate(model=model, dataset=args.dataset)


if __name__ == '__main__':
    coloredlogs.install(level='INFO', fmt='> %(asctime)s %(levelname)-8s %(message)s')
    args = parser.parse_args()

    main(args)
