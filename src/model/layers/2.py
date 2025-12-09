'''
(2): C3k2(
      (cv1): Conv(
        (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
'''

import torch
import torch.nn as nn


class YOLOThirdBlock(nn.Module):
  def __init__(self):
    super().__init__()

    #first convoluation layer
    self.conv0 = nn.conv2d(
      in_channels = 32,
      out_channels = 32,
      kernel_size = 1,
      stride = 1,
      bias = False
    )

    self.bn0 = nn.BatchNorm2d(
      num_features=32,
      eps=1e-3,
      momentum=0.03,
      affine=True,
      track_running_stats=True
    )

    #Second convolution layer
    self.conv1 = nn.conv2d(
      
    )



    self.act = nn.SiLU()


  def foward(self, x):
    x = self.conv0(x)
    x = self.bn0(x)
    x = self.SiLU(x)
    

    

    
