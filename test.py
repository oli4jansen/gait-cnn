import json
import coloredlogs
import torch
from torch.utils.data import DataLoader
import argparse
import logging
from statistics import mean

from tqdm import tqdm

from dataset import GaitDataset
from models import GaitNet

parser = argparse.ArgumentParser(description='Get the test performance of a trained GaitNet model')
parser.add_argument('--checkpoint', type=str, default='checkpoints/full/checkpoint_total.pt')
parser.add_argument('--dataset', type=str, default='data/test/preprocessed')
parser.add_argument('--bs', type=int, default=24)
parser.add_argument('--output', type=str, default='test_results.json')


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    dataset = GaitDataset(args.dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs)

    # Load model and checkpoint
    model = GaitNet(num_classes=len(dataset.classes))
    model.load_state_dict(torch.load(args.checkpoint))
    model.to(device)
    model.eval()

    # Run test and keep track of losses and correct predictions
    test_losses = []
    correct = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            outputs = model(inputs)
            test_losses.append(torch.nn.functional.cross_entropy(outputs, labels).item())
            preds = outputs.argmax(dim=1, keepdim=True)
            correct += preds.eq(labels.view_as(preds)).sum().item()

    # Report and save to file
    accuracy = 100. * correct / len(dataloader.dataset)
    logging.info(f'avg test loss: {mean(test_losses)}')
    logging.info(f'min test loss: {min(test_losses)}')
    logging.info(f'max test loss: {max(test_losses)}')
    logging.info(f'test accuracy: {accuracy}')

    with open(args.output, 'w+') as file:
        json.dump(dict({'test_losses': test_losses, 'test_accuracy': accuracy}), file)


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
