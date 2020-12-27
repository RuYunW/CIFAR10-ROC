import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_NET(torch.nn.Module):
    def __init__(self):
        super(CNN_NET, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3,
                                     out_channels=64,
                                     kernel_size=5,
                                     stride=1,
                                     padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=3,
                                       stride=2)
        self.conv2 = torch.nn.Conv2d(64, 64, 5)
        self.fc1 = torch.nn.Linear(64 * 4 * 4, 384)
        self.fc2 = torch.nn.Linear(384, 192)
        self.fc3 = torch.nn.Linear(192, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
