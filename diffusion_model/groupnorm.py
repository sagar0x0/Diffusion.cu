import torch 
from torch import nn
from torch.nn import functional as F
import groupNorm_kernel


class GroupNorm(nn.Module):
    def __init__(self, num_groups, channels, eps=1e-6):
        super().__init__()
        self.num_groups = num_groups
        self.channels = channels
        self.eps = eps  # added param to make it compatible (redundant)

        # added params
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))


    def forward(self, x):
        input_shape = x.shape

        # x : (Batch_Size, In_Channels, Height, Width)
        Batch_Size, In_Channels, Height, Width = input_shape

        out = torch.zeros_like(x)
        groupNorm_kernel.groupNorm(
            x.data_ptr(), out.data_ptr(), Batch_Size,
            In_Channels, Height, Width, self.num_groups
        )

        # APPLY LEARNABLE PARAMETERS (scale and shift)
        # Reshape weight and bias to (1, channels, 1, 1) for broadcasting
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        
        out = out * weight + bias

        # x: (Batch_Size, In_Channels, Height, Width) -> out: (Batch_Size, In_Channels, Height, Width)
        return out
        






