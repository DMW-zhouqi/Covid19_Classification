import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# image shape flatten
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# Image channel attention mechanism class
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )

    def forward(self, x):
        # [1, 32, 128, 128] => [1, 32, 1, 1]
        avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        # [1, 32, 1, 1] => [1, 32]
        channel_att_raw = self.mlp( avg_pool )
        # [1, 32] => [1, 32, 128, 128]
        scale = F.sigmoid( channel_att_raw ).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

# Image spatial information attention mechanism class
class SpatialGate(nn.Module):
    def __init__(self, gate_channels):
        super(SpatialGate, self).__init__()
        kernel_size = 15
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, gate_channels, kernel_size, stride=1, padding=(kernel_size-1) // 2)

    def forward(self, x):
        # [1, 32, 128, 128] => [1, 2, 128, 128]
        x_compress = self.compress(x)
        # [1, 2, 128, 128] => [1, 32, 128, 128]
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)

        return x * scale

# attention mechanism
class AM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(AM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate(gate_channels)

    def forward(self, x):
        x_out_channel = self.ChannelGate(x)
        x_out_spatial = self.SpatialGate(x)

        return x_out_channel + x_out_spatial

# ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

# AMResNet
class AMResNet(nn.Module):
    def __init__(self):
        super(AMResNet, self).__init__()

        general_features = 32

        # Initial convolution block
        self.init_conv = nn.Conv2d(3, general_features, 3, 1, padding=1)
        self.activation_f = nn.ReLU()

        # ResidualBlock
        self.resblock = ResidualBlock(general_features)

        # Attention Module
        self.attention_module = AM(general_features)

        # Full Connection
        self.FC = nn.Sequential(nn.Linear(32*32*32, 128), 
                            nn.Dropout(0.5),
                            nn.ReLU(),

                            nn.Linear(128, 64),
                            nn.Dropout(0.5),
                            nn.ReLU(),

                            nn.Linear(64, 32),
                            nn.Dropout(0.5),
                            nn.ReLU(),
                            
                            nn.Linear(32, 3))

        # Down sample 1/2
        self.downsample = nn.Conv2d(general_features, general_features, 1, 2)

    def forward(self, x):
        x = self.downsample(self.activation_f(self.init_conv(x))) # [1, 32, 256, 256]
        x = self.downsample(x) # [1, 32, 128, 128]

        x_res = self.resblock(x)
        x_am = self.attention_module(x)

        # add attention module informaton
        x_final = x_am + x_res

        # downsample
        x_final = self.downsample(x_final) # [1, 32, 64, 64]
        x_final = self.downsample(x_final)  # [1, 32, 32, 32]
        x_final = x_final.view(x_final.shape[0], -1)  # Flatten

        x_final = F.softmax(self.FC(x_final), dim=1)
        return x_final


# x = torch.randn(1, 3, 128, 128)
m = AMResNet()
print(summary(m, (3, 512, 512)))
# print(m(x).shape)
