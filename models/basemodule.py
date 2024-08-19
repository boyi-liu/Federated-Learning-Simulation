import torch.cuda

import torch
from torch import nn

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