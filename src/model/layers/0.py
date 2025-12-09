'''
(0): Conv(
      (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(16, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
'''


import torch
import torch.nn as nn


#Defining the block
class YOLOFirstBlock(nn.Module):

    def __init__(self):
        #used to intialize the parent class
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(
            num_features=16,
            eps=1e-3,                    #prevent division by zero
            momentum=0.03,               #used to control or stablize the moving average & variance
            affine=True,                 #enable learnable parameters(scalable parameter & shift parameter)
            track_running_stats=True
        )

        self.act = nn.SiLU()             #activation function
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    


#test working of the layer
model = YOLOFirstLayer()
x = torch.randn(1,3,640,640)
y = model(x)

print("input shape:",x.shape)
print("output shape:",y.shape)