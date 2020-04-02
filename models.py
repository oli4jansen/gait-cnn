import torch
from torchvision.models.video import r2plus1d_18
from stacked_hourglass import HumanPosePredictor, hg2

class GaitNet(torch.nn.Module):
    def __init__(self, num_classes=15):
        super(GaitNet, self).__init__()

        self.r2plus1d_18 = r2plus1d_18(pretrained=True)
        # Simulate identity with empty sequential
        self.r2plus1d_18.fc = torch.nn.Sequential()

        self.pose_model = hg2(pretrained=True)

        # Freeze all pretrained params
        for param in self.r2plus1d_18.parameters():
            param.requires_grad = False

        # Freeze all pretrained params
        for param in self.pose_model.parameters():
            param.requires_grad = False

        self.pose_predictor = HumanPosePredictor(self.pose_model)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=512 + 16 * 2, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, input):
        print('input')
        print(input.size())
        upscaled = torch.nn.functional.interpolate(input, size=224)
        print('upscaled')
        print(upscaled.size())
        joints = self.pose_predictor.estimate_joints(upscaled).view(input.size()[0], input.size()[0][1], -1)
        print('joints')
        print(joints.size())
        cnn_features = self.r2plus1d_18(input)
        print('cnn_features')
        print(cnn_features.size())
        joints_and_cnn = torch.cat(joints, cnn_features, dim=2)
        print('joints_and_cnn')
        print(joints_and_cnn.size())
        return self.classifier(joints_and_cnn)
