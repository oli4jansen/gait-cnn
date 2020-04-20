import torch
import torchvision
from torchvision.models.video import r2plus1d_18
from stacked_hourglass import HumanPosePredictor, hg2
from stacked_hourglass.datasets.mpii import Mpii

FRAMES = 16
CHANNELS = 3
HEIGHT = 112
WIDTH = 112


class Print(torch.nn.Module):
    def __init__(self, string = None):
        super(Print, self).__init__()
        self.string = string

    def forward(self, input):
        if self.string:
            print(f'{self.string}:')
        print(input.size())
        return input

class GaitNet(torch.nn.Module):
    def __init__(self, num_classes=15):
        super(GaitNet, self).__init__()

        self.pose_model = hg2(pretrained=True)
        self.pose_model.to('cpu')
        self.pose_predictor = HumanPosePredictor(self.pose_model)

        self.pose_cnn = torch.nn.Sequential(
            torchvision.models.video.resnet.Conv2Plus1D(16, 64, 32),
            torchvision.models.video.resnet.Conv2Plus1D(64, 256, 128, stride=2),
            torchvision.models.video.resnet.Conv2Plus1D(256, 64, 128, stride=2),
            torch.nn.AdaptiveAvgPool3d((2, 2, 2)),
            torch.nn.Flatten(start_dim=1),
            torch.nn.ReLU(inplace=True)
        )

        self.r2plus1d_18 = r2plus1d_18(pretrained=True)
        # Simulate identity with empty sequential on last fully-connected layer
        self.r2plus1d_18.fc = torch.nn.Sequential()

        # Freeze all pretrained params
        for param in self.pose_model.parameters():
            param.requires_grad = False

        # Freeze all pretrained params
        for param in self.r2plus1d_18.parameters():
            param.requires_grad = False

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=512 + 512, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, input):
        batch_size, channels, frames, height, width = input.size()
        assert(channels == CHANNELS and frames == FRAMES and height == HEIGHT and width == WIDTH)


        input_1 = input.permute(0, 2, 1, 3, 4)
        input_2 = torch.nn.functional.interpolate(input_1, size=[channels, 256, 256])

        # input_2 = torch.size([batch_size, frames, channels, height, width])

        normalized_batch = []
        for images in input_2:
            # images = torch.size([frames, channels, height, width])
            normalized_images = []
            for image in images:
                # image = torch.size([channels, height, width])
                image = torchvision.transforms.functional.normalize(image, Mpii.DATA_INFO.rgb_mean, Mpii.DATA_INFO.rgb_stddev)
                normalized_images.append(torch.unsqueeze(image, 0))
            # normalized_images = [torch.size([channels, height, width])]
            normalized_images = torch.unsqueeze(torch.cat(normalized_images, dim=0), dim=0)
            # normalized_images = torch.size([1, frames, channels, height, width])
            normalized_batch.append(normalized_images)
        # normalized_batch = [torch.size([1, frames, channels, height, width])]

        input_3 = torch.cat(normalized_batch, dim=0)
        # input_3 = torch.size([batch_size, frames, channels, height, width])

        heatmaps_list = [torch.unsqueeze(self.pose_model(video)[-1], dim=0) for video in input_3]

        # Concat tensors in pose list into tensor again
        heatmaps = torch.cat(heatmaps_list, dim=0)
        # torch.size([batch, frames, joints, 64, 64])

        # heatmaps = torch.rand((12, 16, 16, 64, 64))

        pose_cnn_input = heatmaps.permute(0, 2, 1, 3, 4)

        # Run pose CNN on extracted poses
        pose_cnn_output = self.pose_cnn(pose_cnn_input)

        # Run R(2+1)D on the raw pixel data
        cnn_output = self.r2plus1d_18(input)

        # Combine R(2+1)D and pose information for classifier
        classifier_input = torch.cat([cnn_output, pose_cnn_output], dim=1)

        return self.classifier(classifier_input)
