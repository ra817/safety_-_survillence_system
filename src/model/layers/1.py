'''
(1): Conv(
      (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
'''


import torch
import torch.nn as nn


class YOLOSecondBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(
            num_features=32,
            eps=1e-3,
            momentum=0.03,
            affine=True,
            track_running_stats=True
        )

        self.act = nn.SiLU()
    
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x




    

        