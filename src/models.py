import torch
from torch import nn
import torch.nn.functional as F



class LeNet(nn.Module):
    def __init__(self, num_classes, use_log_softmax=False):
        super(LeNet, self).__init__()
        self.use_log_softmax = use_log_softmax
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        if self.use_log_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x
    
    def name(self):
        return "LeNet"
    