import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, args, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.args = args
        self.layer_input = nn.Linear(dim_in, 512)
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, 64)

        self.fc = nn.Linear(64, dim_out)

    def features(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = F.relu(self.layer_input(x))
        x = F.relu(self.layer_hidden1(x))
        x = F.relu(self.layer_hidden2(x))
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def mlp(args, params):
    return MLP(args=args,
               dim_in=params['dim_in'],
               dim_hidden=params['dim_hidden'],
               dim_out=args.class_num)