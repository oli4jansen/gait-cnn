import os
import shutil
import glob
import math
import logging
import subprocess

import coloredlogs

import torch
import torchvision
from PIL import Image
from torch.nn.functional import interpolate
from tqdm import tqdm
from yolov3.yolo import YOLOv3
import numpy as np
import cv2
from stacked_hourglass import HumanPosePredictor, hg2
from stacked_hourglass.datasets.mpii import MPII_JOINT_NAMES

import transforms
from sort import Sort
from utils import group_sequence

EXPECTED_FPS = 30


class Preprocessor():
    def __init__(self):
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

        self.batch_size = 12
        self.detection_threshold = 0.7

        self.detector = YOLOv3(
            device=self.device, img_size=608, person_detector=True, video=True, return_dict=True
        )

        self.tracker = Sort()

        self.pose_model = hg2(pretrained=True)
        self.pose_predictor = HumanPosePredictor(self.pose_model, device=self.device)


    def preprocess_dir(self, input_dir, output_dir):
        logging.info(f'preprocessing {input_dir} -> {output_dir}')

        # Create empty dir
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        videos = glob.glob(os.path.join(input_dir, '*'))
        logging.info(f'{len(videos)} video(s) found')

        for video_path in videos:
            self.preprocess_video(video_path, output_dir)


    def preprocess_video(self, video_path, output_dir):
        # Create a folder to contain the video frames
        # frames_dir = os.path.join(output_dir, os.path.basename(video_path) + '-frames')
        # os.makedirs(frames_dir, exist_ok=True)


        vframes = self.read_video(video_path)

        people = self.find_people(vframes)

        if people is None or len(people) is 0:
            print('no people found')
            # TODO: Clean everything
            # shutil.rmtree()
            return

        person = self.find_most_common_person(people)

        logging.info(f'found person from frames {person["frames"][0]}-{person["frames"][-1]}')

        # Slice only frames with the person in it
        vframes = vframes[person['frames'][0]:person['frames'][-1] + 1]

        assert(len(vframes) == len(person['frames']))


        crops_list = []

        # Crop and resize
        for img, bbox in zip(vframes, person['bbox']):
            crop = self.square_crop(img, bbox)
            crop = torch.unsqueeze(crop, 0)
            crop = interpolate(crop, size=224)
            crops_list.append(crop)

        # crops_list = [interpolate(self.square_crop(img, bbox), 224) for img, bbox in zip(vframes, person['bbox'])]

        crops = torch.Tensor(len(vframes), 3, 224, 224)
        torch.cat(crops_list, out=crops)


        pelvis = self.find_pelvis(crops)

        # img_list = []

        # Create a folder to contain the tracked and cropped video frames
        frames_dir = os.path.join(output_dir, os.path.basename(video_path) + '-frames')
        os.makedirs(frames_dir, exist_ok=True)


        for idx, ((pelvis_x, pelvis_y), img, bbox) in enumerate(zip(pelvis, vframes, person['bbox'])):
            # pelvis_loc is relative to 224x224 crop

            bbox_x, bbox_y = bbox[:2]
            size = bbox[2]

            # Get scale factor of YOLO crop to 224x224
            scale_factor = 224 / size * 1.1

            # Scale the pelvis back to YOLO crop pixel space
            pelvis_x /= scale_factor
            pelvis_y /= scale_factor

            # Get the offset to the top and left of the YOLO crop
            offset_x = bbox_x - (size / 2)
            offset_y = bbox_y - (size / 2)

            # Offset the pelvis with the offset of the YOLO crop (back to original pixel space)
            pelvis_x = int(pelvis_x + offset_x)
            pelvis_y = int(pelvis_y + offset_y)

            img = img[:, pelvis_y - int(size * 0.6):pelvis_y + int(size * 0.6), pelvis_x - int(size * 0.6):pelvis_x + int(size * 0.6)]

            img = torch.unsqueeze(img, 0)
            img = interpolate(img, size=112)

            cv2.imwrite(os.path.join(frames_dir, f'{idx:06d}.png'), img)

        command = [
            'ffmpeg', '-y', '-threads', '16', '-i', os.path.join(frames_dir, f'{idx:06d}.png'), '-profile:v', 'baseline',
            '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', os.path.basename(video_path),
        ]

        # print(f'Running \"{" ".join(command)}\"')
        subprocess.call(command)

        # images_to_video(img_folder=tmp_write_folder, output_vid_file=output_file)
        # shutil.rmtree(tmp_write_folder)


    def square_crop(self, frame, bbox):
        """ Frame is of shape (channels, width, height) """

        center_x, center_y = bbox[:2]
        size = bbox[3] * 0.6

        return frame[:, int(center_y - size):int(center_y + size), int(center_x - size):int(center_x + size)]

    def read_video(self, video_path):
        """ Takes path to a video and returns tensor with the shape (num_frames, height, width, channels). """

        logging.info(f'loading video {video_path}')
        vframes, aframes, info = torchvision.io.read_video(video_path, pts_unit='sec')

        # Check if video fps matches expectation
        video_fps = info['video_fps']
        if not math.isclose(info['video_fps'], EXPECTED_FPS, abs_tol=0.25):
            raise ValueError(f'expected video fps {EXPECTED_FPS} but got {video_fps}')

        return vframes.permute(0, 3, 1, 2).to(torch.float32) / 255


    def find_people(self, vframes):
        logging.info('running YOLOv3 multi-people tracker')

        dataloader = torch.utils.data.DataLoader(vframes, batch_size=self.batch_size, num_workers=0)

        # initialize tracker
        self.tracker = Sort()

        trackers = []
        for batch in tqdm(dataloader):
            batch = batch.to(self.device)

            predictions = self.detector(batch)

            for pred in predictions:
                bb = pred['boxes'].cpu().numpy()
                sc = pred['scores'].cpu().numpy()[..., None]
                dets = np.hstack([bb, sc])
                dets = dets[sc[:, 0] > self.detection_threshold]

                # if nothing detected do not update the tracker
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
                c_x, c_y = d[0] + w/2, d[1] + h/2
                w = h = np.where(w / h > 1, w, h)
                bbox = np.array([c_x, c_y, w, h])

                if person_id in people.keys():
                    people[person_id]['bbox'].append(bbox)
                    people[person_id]['frames'].append(frame_idx)
                else:
                    people[person_id] = {
                        'bbox' : [],
                        'frames' : [],
                    }
                    people[person_id]['bbox'].append(bbox)
                    people[person_id]['frames'].append(frame_idx)

        for k in people.keys():
            people[k]['bbox'] = np.array(people[k]['bbox']).reshape((len(people[k]['bbox']), 4))
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
        # crops = torch.nn.functional.interpolate(frames, size=224)
        #
        # print(crops.size())

        joints_frames = self.pose_predictor.estimate_joints(frames, flip=True)

        print(joints_frames.size())

        return [joints[MPII_JOINT_NAMES.index('pelvis')] for joints in joints_frames]



if __name__ == '__main__':
    coloredlogs.install(level='INFO', fmt='> %(asctime)s %(levelname)-8s %(message)s')

    # Preprocessor().preprocess_dir('/Users/o.f.jansen/Desktop/single_person_videos/small', 'tmp')
    Preprocessor().preprocess_dir('/home/olivier/single_person_videos/violence', 'tmp')
