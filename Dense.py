import torch
import torch.nn as nn
from utils import ConvBlock

class DenseLayer(nn.Module):
    def __init__(self, num_channels, growth):
        super(DenseLayer, self).__init__()
        self.conv = ConvBlock(num_channels, growth, kernel_size=3, act_type='lrelu', norm_type=None)

    def forward(self, x):
        out = self.conv(x)
        out = torch.cat((x, out), 1)
        return out

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.in_channels  = 1
        self.out_channels = 1
        self.num_layers   = 5
        self.num_channels = 2 * self.in_channels#in_channels
        self.num_features = 44#num_features
        self.growth = 44#channel growth in dense blocks
        modules = []
        self.conv_1 = ConvBlock(self.num_channels, self.num_features, kernel_size=3, act_type='lrelu', norm_type=None)
        for i in range(self.num_layers):#number of dense layers
            modules.append(DenseLayer(self.num_features, self.growth))
            self.num_features += self.growth
        self.dense_layers = nn.Sequential(*modules)#编码器
        self.sub = nn.Sequential(ConvBlock(self.num_features, 128, kernel_size=3, act_type='lrelu', norm_type=None),#解码器
                                 ConvBlock(128, 64, kernel_size=3, act_type='lrelu', norm_type=None),
                                 ConvBlock(64, 32, kernel_size=3, act_type='lrelu', norm_type=None),
                                 nn.Conv2d(32, self.out_channels, kernel_size=3, stride=1, padding=1),#只有Y这个通道，所以输出通道为1
                                 nn.Tanh())

    def forward(self, visible, lwir):
        x = torch.cat((visible, lwir), dim=1)
        x = self.conv_1(x)
        x = self.dense_layers(x)
        x = self.sub(x)
        return x