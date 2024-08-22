import torch.nn as nn
import torch.nn.functional as F


class CNNCifar(nn.Module):
    def __init__(self, args, dim_out):
        super(CNNCifar, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)

        self.fc = nn.Linear(192, dim_out)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def features(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def logits(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


def cnncifar(args, params):
    return CNNCifar(args=args, dim_out=args.class_num)
