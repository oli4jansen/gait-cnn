import os
import shutil
import glob
import math
import logging
import subprocess

import coloredlogs

import torch
import torchvision
from torch.nn.functional import interpolate
from yolov3.yolo import YOLOv3
import numpy as np
import cv2
from stacked_hourglass import HumanPosePredictor, hg2
from stacked_hourglass.datasets.mpii import MPII_JOINT_NAMES

from sort import Sort
from utils import group_sequence


EXPECTED_FPS = 24

coloredlogs.install(level='INFO', fmt='> %(asctime)s %(levelname)-8s %(message)s')

class Preprocessor():
    def __init__(self, output_dir, num_frames_per_clip=16, distance_between_clips=-4):
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(torch.cuda.current_device())
            logging.info('using %s', device_name)
            # Force CUDA tensors by default
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.device = 'cuda'
        else:
            logging.info('using CPU, limiting threads')
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            self.device = 'cpu'

        self.output_dir = output_dir

        if distance_between_clips <= -1 * num_frames_per_clip:
            raise ValueError('distance_between_clips cannot be equal to or less than -1 * num_frames_per_clip')

        self.num_frames_per_clip = num_frames_per_clip
        self.distance_between_clips = distance_between_clips

        self.detector = YOLOv3(
            device=self.device, img_size=608, person_detector=True, video=True, return_dict=True
        )

        self.tracker = Sort()

        self.pose_model = hg2(pretrained=True)
        self.pose_predictor = HumanPosePredictor(self.pose_model, device=self.device)

    def preprocess_dir(self, input_dir):
        logging.info(f'preprocessing {input_dir} -> {self.output_dir}')

        # Create empty dir
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        videos = glob.glob(os.path.join(input_dir, '*'))
        logging.info(f'{len(videos)} video(s) found')

        for video_path in videos:
            self.preprocess_video(video_path)

    def preprocess_video(self, video_path):

        vframes = self.read_video(video_path)
        people = self.find_people(vframes)

        if people is None or len(people) is 0:
            logging.warming(f'no people found in video {video_path}')
            return

        # Find the most common person of all people
        person = self.find_most_common_person(people)

        # Slice only frames with the person in it
        vframes = vframes[person['frames'][0]:person['frames'][-1] + 1]

        # Crop and resize frames according to YOLO bboxes from most common person
        crops_list = []
        for img, bbox in zip(vframes, person['bbox']):
            crop = self.square_crop(img, *bbox)
            crop = torch.unsqueeze(crop, 0)
            # Resize to 224x224 is required by stacked hourglass that finds pelvis
            crop = interpolate(crop, size=224)
            crops_list.append(crop)

        # YOLO crops to tensor
        crops = torch.Tensor(len(vframes), 3, 224, 224)
        torch.cat(crops_list, out=crops)

        # Find the pelvis in each of the crops
        pelvis_locations = self.find_pelvis(crops)

        # Create a folder to contain the tracked and cropped video frames
        frames_dir = os.path.join(self.output_dir, os.path.basename(video_path) + '-frames')
        os.makedirs(frames_dir, exist_ok=True)

        for idx, ((pelvis_x, pelvis_y), img, bbox) in enumerate(zip(pelvis_locations, vframes, person['bbox'])):
            # Pelvis location is relative to the 224x224 YOLO crop
            # It needs to be sized back and an offset needs to be added to center the original image around it

            bbox_x, bbox_y = bbox[:2]
            size = bbox[2]

            # Get scale factor of YOLO crop to 224x224
            scale_factor = 224 / size

            # Scale the pelvis back to YOLO crop pixel space
            pelvis_x /= scale_factor
            pelvis_y /= scale_factor

            # Get the offset to the top and left of the YOLO crop
            offset_x = bbox_x - (size / 2)
            offset_y = bbox_y - (size / 2)

            # Offset the pelvis with the offset of the YOLO crop (back to original pixel space)
            pelvis_x = int(pelvis_x + offset_x)
            pelvis_y = int(pelvis_y + offset_y)

            img = self.square_crop(img, pelvis_x, pelvis_y, size / 1.15)

            img = torch.unsqueeze(img, 0)
            img = interpolate(img, size=112)
            img = torch.squeeze(img, 0)

            img = np.clip(img.numpy() * 255, 0, 255)
            img = img.astype(np.uint8)
            img = img.transpose(1, 2, 0)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(frames_dir, f'{idx:06d}.png'), img)

        clips = []
        frames_left = len(vframes)
        while frames_left > self.num_frames_per_clip:
            start = len(vframes) - frames_left
            end = start + self.num_frames_per_clip
            clips.append((start, end))
            frames_left -= self.num_frames_per_clip
            frames_left -= self.distance_between_clips


        # Combine frames into video
        images_path = os.path.join(frames_dir, '%06d.png')

        for idx, (start, end) in enumerate(clips):
            output_path = os.path.join(self.output_dir, f'clip-{idx:04d}_' + os.path.basename(video_path))

            command = [
                'ffmpeg', '-start_number', str(start), '-y', '-threads', '16', '-i', images_path,
                '-profile:v', 'baseline', '-vframes', str(int(end - start)), '-level', '3.0', '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_path,
            ]

            subprocess.call(command)

        # Clear frames directory
        shutil.rmtree(frames_dir)

        logging.info(f'video saved at {output_path}')

    def square_crop(self, frame, center_x, center_y, size):
        """ Input frame must be of shape (channels, width, height) """

        y_start = max(0, int(center_y - size))
        y_end = min(int(center_y + size), frame.size()[1])
        x_start = max(0, int(center_x - size))
        x_end = min(int(center_x + size), frame.size()[2])

        return frame[:, y_start:y_end, x_start:x_end]

    def read_video(self, video_path):
        """ Takes path to a video and returns tensor with the shape (num_frames, height, width, channels). """

        # Load the video directly as Tensor
        vframes, aframes, info = torchvision.io.read_video(video_path, pts_unit='sec')

        # Check if video fps matches expectation
        video_fps = info['video_fps']
        if not math.isclose(info['video_fps'], EXPECTED_FPS, abs_tol=0.25):
            raise ValueError(f'expected video fps {EXPECTED_FPS} but got {video_fps}')

        # Permute into shape (num_frames, channels, height, width) and map to [0-1] range
        return vframes.permute(0, 3, 1, 2).to(torch.float32) / 255

    def find_people(self, frames):
        logging.info('running YOLOv3 multi-people tracker')

        dataloader = torch.utils.data.DataLoader(frames, batch_size=16, num_workers=0)

        self.tracker = Sort()
        detection_threshold = 0.7

        trackers = []
        for batch in dataloader:
            batch = batch.to(self.device)

            predictions = self.detector(batch)

            for pred in predictions:
                bb = pred['boxes'].cpu().numpy()
                sc = pred['scores'].cpu().numpy()[..., None]
                dets = np.hstack([bb, sc])
                dets = dets[sc[:, 0] > detection_threshold]

                if dets.shape[0] > 0:
                    track_bbs_ids = self.tracker.update(dets)
                else:
                    track_bbs_ids = np.empty((0, 5))
                trackers.append(track_bbs_ids)

        people = dict()
        for frame_idx, tracks in enumerate(trackers):
            for d in tracks:
                person_id = int(d[4])

                w, h = d[2] - d[0], d[3] - d[1]
                c_x, c_y = d[0] + w / 2, d[1] + h / 2
                size = np.where(w / h > 1, w, h)
                bbox = np.array([c_x, c_y, size * 0.6])

                if person_id in people.keys():
                    people[person_id]['bbox'].append(bbox)
                    people[person_id]['frames'].append(frame_idx)
                else:
                    people[person_id] = {
                        'bbox': [],
                        'frames': [],
                    }
                    people[person_id]['bbox'].append(bbox)
                    people[person_id]['frames'].append(frame_idx)

        for k in people.keys():
            people[k]['bbox'] = np.array(people[k]['bbox']).reshape((len(people[k]['bbox']), 3))
            people[k]['frames'] = np.array(people[k]['frames'])

        return people

    def find_most_common_person(self, people):
        if people is None or len(people) == 0:
            return None

        person = None
        # Loop over all detected people
        for p in people.values():
            # Group the frames on which the person was detected into groups
            p['frames'] = group_sequence(p['frames'])
            # If the largest group of this person is larger than the largest found thus far, it is saved
            if person is None or max([len(g) for g in p['frames']]) > max([len(g) for g in person['frames']]):
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

        return person

    def find_pelvis(self, frames):
        joints_frames = self.pose_predictor.estimate_joints(frames, flip=True)
        return [joints[MPII_JOINT_NAMES.index('pelvis')] for joints in joints_frames]

if __name__ == '__main__':
    Preprocessor(output_dir='data/synth-cmu-clips').preprocess_dir('data/synth-cmu')
