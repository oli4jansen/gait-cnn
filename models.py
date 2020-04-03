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

        batch_size, channels, frames, height, width = input.size()

        joints_input = input.permute(0, 2, 1, 3, 4)
        joints_input = torch.nn.functional.interpolate(joints_input, size=[channels, 224, 224])

        print('upscaled')
        print(joints_input.size())

        joints_list = [self.pose_predictor.estimate_joints(i, flip=True) for i in joints_input]
        joints_output = torch.Tensor(batch_size, frames, 16, 2)
        torch.cat(joints_list, dim=0, out=joints_output)

        print('joints_output')
        print(joints_output.size())
        joints_output = joints_output.view(size=(batch_size, frames * 16 * 2))
        print(joints_output.size())

        cnn_output = self.r2plus1d_18(input)

        print('cnn_output')
        print(cnn_output.size())

        classifier_input = torch.Tensor(batch_size, cnn_output.size()[1] + joints_output.size()[1])
        torch.cat([cnn_output, joints_output], dim=1, out=classifier_input)

        print('classifier_input')
        print(classifier_input.size())
        return self.classifier(classifier_input)
