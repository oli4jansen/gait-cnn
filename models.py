import torch
from torchvision.models.video import r2plus1d_18

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GaitNet(torch.nn.Module):
    def __init__(self):
        super(GaitNet, self).__init__()

        self.r2plus1d_18 = r2plus1d_18(pretrained=True)

        # Freeze all pretrained params
        for param in self.r2plus1d_18.parameters():

            param.requires_grad = False

        print(self.r2plus1d_18.fc)
        # Replace FC with our layer and enable gradients
        # self.r2plus1d_18.fc = Identity()
        self.r2plus1d_18.fc = torch.nn.Linear(in_features=512, out_features=512)

        # for param in self.r2plus1d_18.fc.parameters():
        #     param.requires_grad = True

    def forward(self, input):
        return self.r2plus1d_18(input)


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x