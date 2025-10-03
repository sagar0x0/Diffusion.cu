import torch
from torch import nn
import Conv2d_cuda_kernel


class Conv2d_K3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert kernel_size == 3, "kernel_size should be 3"
        self.kernel_size = kernel_size
        self.padding = padding

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

        # Initialize parameters properly
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)


    def forward(self, x):
        input_shape = x.shape

        # x: (Batch_Size, In_Channels, Height, Width)
        Batch_Size, In_Channels, Height, Width = input_shape

        # Apply padding if needed
        if self.padding > 0:
            x = torch.nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)

        out = torch.zeros((Batch_Size, self.out_channels, Height, Width),
                          dtype= torch.float32, device='cuda')
        
        Conv2d_cuda_kernel.Conv2d_k3(
            x.data_ptr(), self.weight.data_ptr(), out.data_ptr(),
            Batch_Size, self.in_channels, self.out_channels, Height, Width
        )

        # ADD BIAS IF PRESENT
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
            
        return out
