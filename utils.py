import glob
import subprocess
import os
import cv2
import torch
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_tensor


def video_to_images(video_file, image_folder):
    """Split the video into frame images using ffmpeg"""

    command = ['ffmpeg',
               '-i', video_file,
               '-f', 'image2',
               '-v', 'error',
               '-r', '24',
               f'{image_folder}/%06d.png']

    subprocess.call(command, stdout=subprocess.DEVNULL)

def group_sequence(lst):
    res = [[lst[0]]]
    for i in range(1, len(lst)):
        if lst[i - 1] + 1 == lst[i]:
            res[-1].append(lst[i])
        else:
            res.append([lst[i]])
    return res

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

