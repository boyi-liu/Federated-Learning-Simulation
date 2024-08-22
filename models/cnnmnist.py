import torch.nn as nn
import torch.nn.functional as F

class CNNMnist(nn.Module):
    def __init__(self, args, dim_out):
        super(CNNMnist, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 64)

        self.fc = nn.Linear(64, dim_out)

    def features(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(F.dropout2d(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def cnnmnist(args, params):
    return CNNMnist(args=args, dim_out=args.class_num)