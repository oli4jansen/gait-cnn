import glob
import os
import shutil
import operator

import cv2
import numpy as np
import torch
from multi_person_tracker import MPT
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import list_dir
from torchvision.datasets.video_utils import VideoClips

from utils import video_to_images

mpt = MPT(
    detector_type='yolo',
    batch_size=10,
    yolo_img_size=416,
    output_format='dict'
)


def group_sequence(lst):
    res = [[lst[0]]]
    for i in range(1, len(lst)):
        if lst[i - 1] + 1 == lst[i]:
            res[-1].append(lst[i])
        else:
            res.append([lst[i]])
    return res

def preprocess_directory(video_dir, dataset_dir='data'):
    # Create empty dir
    if os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)

    videos = glob.glob(os.path.join(video_dir, '*'))
    for video in videos:

        # Create a folder to contain the video frames
        frames_dir = os.path.join(dataset_dir, os.path.basename(video) + '-frames')
        os.makedirs(frames_dir, exist_ok=True)

        # Create a folder to contain the tracked and cropped video frames
        crops_dir = os.path.join(dataset_dir, os.path.basename(video) + '-crops')
        os.makedirs(crops_dir, exist_ok=True)

        # Convert video to images
        video_to_images(video, frames_dir)

        # Run multi-person tracker to find person bounding boxes in video frames
        people = mpt(frames_dir)


        if len(people) == 0:
            print(f'no people found in video {video}')
            shutil.rmtree(frames_dir)
            continue

        person = None
        # Loop over all detected people
        for p in people.values():
            # Group the frames on which the person was detected into groups
            p['frames'] = group_sequence(p['frames'])
            # If the largest group of this person is larger than the largest found thus far, it is saved
            if max([len(g) for g in p['frames']]) > max([len(g) for g in person['frames']]):
                person = p

        # Find the index of the largest frames group
        index = np.argmax([len(g) for g in person['frames']])
        # Sum the number of frames in groups before the largest group
        start_count = sum([len(g) for g in person['frames'][:index]])
        # Sum the number of frames in groups after the largest group
        end_count = sum([len(g) for g in person['frames'][index + 1:]])

        # Slice the number of bounding boxes and frames
        if end_count > 0:
            person['bbox'] = person['bbox'][start_count:-end_count]
        else:
            person['bbox'] = person['bbox'][start_count:]

        person['frames'] = person['frames'][index]

        # Loop over frames with tracks and images
        for frame, img in enumerate(ImageFolder(frames_dir)):
            if frame not in person['frames']:
                print(f'person not detected in frame {frame}')
            else:
                offset = np.where(person['frames'] == frame)[0][0]
                center_x, center_y = person['bbox'][offset][:2]
                size = person['bbox'][offset][3] * 0.6

                crop_img = img[int(center_y - size):int(center_y + size), int(center_x - size):int(center_x + size)]
                crop_img = cv2.resize(crop_img, (224, 224))
                crop_filename = os.path.join(crops_dir, f"{frame:06d}.png")
                cv2.imwrite(crop_filename, crop_img)

        shutil.rmtree(frames_dir)


class ImageFolder(Dataset):
    def __init__(self, image_folder):
        self.image_file_names = glob.glob(os.path.join(image_folder, '*.png'))
        self.image_file_names = sorted(self.image_file_names)

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        return cv2.imread(self.image_file_names[idx])


class VideoDataset(Dataset):

    def __init__(self, dataset_dir):
        self.video_names = [n.replace('-crops', '') for n in glob.glob(os.path.join(dataset_dir, '*-crops'))]

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video = self.video_names[idx]

        # subject = video.split('_')[2]

        image_file_names = glob.glob(os.path.join(video + '-crops', '*.png'))
        image_file_names = sorted(image_file_names)
        crops = np.array([cv2.imread(img) for img in image_file_names])

        return crops




def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if extensions is None or path.lower().endswith(extensions):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

class Kinetics400(VisionDataset):
    """
    `Kinetics-400 <https://deepmind.com/research/open-source/open-source-datasets/kinetics/>`_
    dataset.

    Kinetics-400 is an action recognition video dataset.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.

    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.

    Internally, it uses a VideoClips object to handle clip creation.

    Args:
        root (string): Root directory of the Kinetics-400 Dataset.
        frames_per_clip (int): number of frames in a clip
        step_between_clips (int): number of frames between each clip
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        label (int): class of the video clip
    """

    def __init__(self, root, frames_per_clip, step_between_clips=1, frame_rate=None,
                 extensions=('avi','mp4',), transform=None, _precomputed_metadata=None,
                 num_workers=1, _video_width=0, _video_height=0,
                 _video_min_dimension=0, _audio_samples=0, _audio_channels=0):
        super(Kinetics400, self).__init__(root)

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
            _audio_channels=_audio_channels,
        )
        self.transform = transform

    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label