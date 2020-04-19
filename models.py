import torch
import torchvision
from torchvision.models.video import r2plus1d_18
from stacked_hourglass import HumanPosePredictor, hg2

FRAMES = 16
CHANNELS = 3
HEIGHT = 112
WIDTH = 112


class Print(torch.nn.Module):
    def forward(self, input):
        print(input.size())
        return input

class GaitNet(torch.nn.Module):
    def __init__(self, num_classes=15):
        super(GaitNet, self).__init__()

        self.pose_model = hg2(pretrained=True)
        self.pose_predictor = HumanPosePredictor(self.pose_model)

        self.pose_cnn = torch.nn.Sequential(
            torchvision.models.video.resnet.Conv2Plus1D(1, 64, 32, padding=2),
            torchvision.models.video.resnet.Conv2Plus1D(64, 16, 32),
            torch.nn.MaxPool3d((3, 3, 1)),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(in_features=2304, out_features=num_classes),
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
            torch.nn.Linear(in_features=512 + num_classes, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=num_classes)
        )

    def forward(self, input):
        batch_size, channels, frames, height, width = input.size()
        assert(channels == CHANNELS and frames == FRAMES and height == HEIGHT and width == WIDTH)

        # Swap channels and frames and upsize to 224x224 for stacked hourglass pose estimator
        joints_input = input.permute(0, 2, 1, 3, 4)
        # TODO: check torch.nn.Upsample
        joints_input = torch.nn.functional.interpolate(joints_input, size=[channels, 224, 224])
        # Estimate joints for each sample in batch (pose estimator is implemented for images so video is already batch)
        pose_list = [torch.unsqueeze(self.pose_predictor.estimate_joints(i, flip=True), 0) for i in joints_input]

        # Concat tensors in pose list into tensor again
        pose_cnn_input = torch.cat(pose_list, dim=0)
        # Add an empty channels dimension
        pose_cnn_input = torch.unsqueeze(pose_cnn_input, 1)

        # pose_cnn_input = torch.rand(size=(batch_size, 1, frames, 16, 2))

        # Run pose CNN on extracted poses
        pose_cnn_output = self.pose_cnn(pose_cnn_input)

        # Run R(2+1)D on the raw pixel data
        cnn_output = self.r2plus1d_18(input)

        # Combine R(2+1)D and pose information for classifier
        classifier_input = torch.cat([cnn_output, pose_cnn_output], dim=1)

        return self.classifier(classifier_input)
