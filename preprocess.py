import os
import shutil
import glob
import math
import logging
import coloredlogs

import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from yolov3.yolo import YOLOv3
import numpy as np
import cv2
from stacked_hourglass import HumanPosePredictor, hg2

import transforms
from sort import Sort
from utils import group_sequence

EXPECTED_FPS = 30


class Preprocessor():
    def __init__(self, device=None):
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(torch.cuda.current_device())
            logging.info('using %s', device_name)
            # Force CUDA tensors by default
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
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
        frames_dir = os.path.join(output_dir, os.path.basename(video_path) + '-frames')
        os.makedirs(frames_dir, exist_ok=True)

        # Create a folder to contain the tracked and cropped video frames
        crops_dir = os.path.join(output_dir, os.path.basename(video_path) + '-crops')
        os.makedirs(crops_dir, exist_ok=True)

        vframes = self.read_video(video_path)

        people = self.find_people(vframes)

        if people is None or len(people) is 0:
            print('no people found')
            # TODO: Clean everything
            # shutil.rmtree()
            return

        person = self.find_most_common_person(people)

        logging.info(f'found person from frames {person["frames"][0]}-{person["frames"][-1]}')


        # cframes = torch.tensor([])
        # Loop over frames with tracks and images
        for frame, img in enumerate(vframes):
            if frame in person['frames']:
                offset = np.where(person['frames'] == frame)[0][0]
                center_x, center_y = person['bbox'][offset][:2]
                size = person['bbox'][offset][3] * 0.6

                yolo_crop = img[:,int(center_y - size):int(center_y + size), int(center_x - size):int(center_x + size)]

                # tensor([], size=(0, 367, 1920))
                yolo_crop = yolo_crop.permute(1, 2, 0)

                print(yolo_crop.shape)

                yolo_crop = cv2.resize(yolo_crop, (224, 224))

                joints = self.pose_predictor.estimate_joints(yolo_crop, flip=True)

                print(joints)

                # cframes = torch.cat(cframes, crop)


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

        dataloader = torch.utils.data.DataLoader(vframes, batch_size=self.batch_size, num_workers=8)

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
            print(p)
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




# def find_pelvis(video):
#     pass


if __name__ == '__main__':
    coloredlogs.install(level='INFO', fmt='> %(asctime)s %(levelname)-8s %(message)s')

    Preprocessor(device='cpu').preprocess_dir('/Users/o.f.jansen/Desktop/single_person_videos/small', 'tmp')
