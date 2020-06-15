import json
import coloredlogs
import numpy as np
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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
    acc1_list = np.array([])
    acc5_list = np.array([])
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            outputs = model(inputs)
            test_losses.append(torch.nn.functional.cross_entropy(outputs, labels).item())

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            acc1_list.append(acc1)
            acc5_list.append(acc5)


    # Report and save to file
    acc1 = np.mean(acc1_list)
    acc5 = np.mean(acc5_list)
    logging.info(f'avg test loss: {mean(test_losses)}')
    logging.info(f'min test loss: {min(test_losses)}')
    logging.info(f'max test loss: {max(test_losses)}')
    logging.info(f'test accuracy (top-1): {acc1}')
    logging.info(f'test accuracy (top-5): {acc5}')

    with open(args.output, 'w+') as file:
        json.dump(dict({'test_losses': test_losses, 'accuracy-top-1': acc1, 'accuracy-top-5': acc5}), file)



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
