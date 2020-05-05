import os
import random
from glob import glob

import torch
import torchvision
from torchvision.datasets import VisionDataset


class GaitDataset(VisionDataset):
    def __init__(self, root, for_training=False, limit=None):
        super(GaitDataset, self).__init__(root)

        self.videos, self.id_to_class, self.class_counts = parse_synth_folder(root, limit=limit)
        self.classes = sorted(self.id_to_class.values())

        if for_training:
            self.transform = torchvision.transforms.Compose([
                to_normalized_float_tensor,
                resize,
                random_horizontal_flip,
                normalize
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                to_normalized_float_tensor,
                resize,
                normalize
            ])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        path, class_id = self.videos[idx]
        vframes, aframes, info = torchvision.io.read_video(path, pts_unit='sec')

        if self.transform is not None:
            vframes = self.transform(vframes)

        return vframes, class_id


def parse_synth_folder(dir, limit=None, extensions=['mp4']):
    """ Parses SYDAMO folders for GaitDataset creation """

    videos = []
    class_to_id = dict()
    class_counts = []

    dir = os.path.expanduser(dir)
    file_list = []
    if extensions is None:
        file_list += glob(os.path.join(dir, '*'))
    else:
        for extension in extensions:
            file_list += glob(os.path.join(dir, f'*.{extension}'))

    if limit is not None and int(limit) < len(file_list):
        file_list = sorted(file_list, key=lambda x: int(os.path.basename(x).split('_')[0]))[:int(limit)]

    for path in file_list:
        if os.path.isdir(path):
            continue
        filename = os.path.basename(path)
        # Extract class from filename, this is very specific code
        _class = '_'.join(filename.split('_')[1:-2])
        if _class not in class_to_id:
            class_id = len(class_to_id)
            class_to_id[_class] = class_id
            class_counts.append(0)

        item = (path, class_to_id[_class])
        videos.append(item)
        class_counts[item[1]] += 1

    id_to_class = {v: k for k, v in class_to_id.items()}

    return videos, id_to_class, class_counts


def random_horizontal_flip(vid):
    if random.random() < 0.5:
        return vid.flip(dims=(-1,))
    return vid


def resize(vid):
    size = (112, 112)
    # NOTE: using bilinear interpolation because we don't work on minibatches at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid, size=size, scale_factor=scale, mode='bilinear', align_corners=False)


def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255


def normalize(vid):
    # Normalization equals Kinetics dataset
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]
    shape = (-1,) + (1,) * (vid.dim() - 1)
    mean = torch.as_tensor(mean).reshape(shape)
    std = torch.as_tensor(std).reshape(shape)
    return (vid - mean) / std
