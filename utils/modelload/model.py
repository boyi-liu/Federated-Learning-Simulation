import torch.cuda

import torch
from torch import nn
import torch.nn.functional as F

class BaseModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parameters_to_tensor(self, local_params=None):
        params = []
        for idx, (name, param) in enumerate(self.named_parameters()):
            # NOTE: only send global params
            if local_params is None or local_params[idx] is False:
                params.append(param.view(-1))
        params = torch.cat(params, 0)
        return torch.nan_to_num(params, nan=0.0, posinf=0.0, neginf=0.0)

    def tensor_to_parameters(self, tensor, local_params=None):
        param_index = 0
        for idx, (name, param) in enumerate(self.named_parameters()):
            if local_params is None or local_params[idx] is False:
                # === get shape & total size ===
                shape = param.shape
                param_size = 1
                for s in shape:
                    param_size *= s

                # === put value into param ===
                # .clone() is a deep copy here
                param.data = tensor[param_index: param_index+param_size].view(shape).detach().clone()
                param_index += param_size

class CNNCifar(BaseModule):
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

class MLP(BaseModule):
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

class CNNMnist(BaseModule):
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