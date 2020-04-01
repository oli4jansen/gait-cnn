import os
from glob import glob

import torchvision
from torchvision.datasets import VisionDataset



def parse_synth_folder(dir, extensions=None):
    """ Function parses SYDAMO folders for Dataset creation """

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

    for path in file_list:
        if os.path.isdir(path):
            continue
        filename = os.path.basename(path)
        # Extract class from filename
        _class = '_'.join(filename.split('_')[1:-1])
        if _class not in class_to_id:
            class_id = len(class_to_id)
            class_to_id[_class] = class_id
            class_counts[class_id] = 0

        item = (path, class_to_id[_class])
        videos.append(item)
        class_counts[item[1]] += 1

    id_to_class = {v: k for k, v in class_to_id.items()}

    return videos, id_to_class, class_counts

class GaitDataset(VisionDataset):

    def __init__(self, root, transform=None):
        super(GaitDataset, self).__init__(root)

        self.videos, self.id_to_class, self.class_counts = parse_synth_folder(root)
        self.classes = sorted(self.id_to_class.values())
        self.transform = transform

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        path, class_id = self.videos[idx]
        vframes, aframes, info = torchvision.io.read_video(path)

        if self.transform is not None:
            vframes = self.transform(vframes)

        return vframes, class_id
