import torch.nn as nn
import torch
from torch.nn import functional as F


class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
      
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)


    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        x = torch.sigmoid(x) * residual #
        
        return x



class Spatial_Att(nn.Module):
    def __init__(self, shape):
        super(Spatial_Att, self).__init__()
        self.H=shape
        self.W=shape
        self.bn=nn.BatchNorm1d(self.H*self.W,affine=True)
        

    def forward(self, x):
        batch,channel,H,W=x.size()

        residual = x
        
        x=x.view(batch,channel,-1)
        x = x.permute(0, 2, 1).contiguous()
        x=self.bn(x)
        x = x.permute(0, 2, 1).contiguous()
        weight_bn = self.bn.weight.data.abs() / torch.sum(self.bn.weight.data.abs())
        x = weight_bn* x
        x=x.view(batch, channel, H, W)

        x = torch.sigmoid(x) * residual #

        return x


class Att(nn.Module):
    def __init__(self, channels,shape, out_channels=None, no_spatial=True):
        super(Att, self).__init__()
        self.Channel_Att = Channel_Att(channels)
        self.no_spatial = no_spatial
        self.Spatial_Att = Spatial_Att(shape)

    def forward(self, x):
        x_out1=self.Channel_Att(x)
        if not self.no_spatial:
            x_out2 = self.Spatial_Att(x_out1)
        else:
            x_out2 = x_out1
        return x_out2  
