import torch
from torchvision.models.video import r2plus1d_18

class GaitNet(torch.nn.Module):
    def __init__(self, num_classes=15):
        super(GaitNet, self).__init__()

        self.r2plus1d_18 = r2plus1d_18(pretrained=True)

        # Freeze all pretrained params
        for param in self.r2plus1d_18.parameters():
            param.requires_grad = False

        # Replace FC with our layer and enable gradients
        # self.r2plus1d_18.fc = torch.nn.Linear(in_features=512, out_features=512)

        self.r2plus1d.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.r2plus1d_18.fc.in_features, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, input):
        return self.r2plus1d_18(input)
