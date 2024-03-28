# -*- coding:utf-8 -*-
from ccsdg.models.unet import UnetBlock, SaveFeatures
from ccsdg.models.resnet import resnet34, resnet18, resnet50, resnet101, resnet152
from torch import nn
import torch
import torch.nn.functional as F


class Projector(nn.Module):
    def __init__(self, output_size=1024):
        super(Projector, self).__init__()
        self.conv = nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(131072, output_size)

    def forward(self, x_in):
        x = self.conv(x_in)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x


class UNetCCSDG(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False):
        super().__init__()
        cut, lr_cut = [8, 6]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        layers = list(base_model(pretrained=pretrained).children())[:cut]
        first_layer = layers[0]
        other_layers = layers[1:]
        base_layers = nn.Sequential(*other_layers)
        self.first_layer = first_layer
        self.rn = base_layers

        self.channel_prompt = nn.Parameter(torch.randn(2, 64, 1, 1))

        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [1, 3, 4, 5]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)

    def forward_first_layer(self, x, tau=0.1):
        x = self.first_layer(x)  # 8 64 256 256

        channel_prompt_onehot = torch.softmax(self.channel_prompt/tau, dim=0)
        f_content = x * channel_prompt_onehot[0].view(1, *channel_prompt_onehot[0].shape)
        f_style = x * channel_prompt_onehot[1].view(1, *channel_prompt_onehot[1].shape)
        return f_content, f_style

    def forward(self, x, tau=0.1):
        x = self.first_layer(x)  # 8 64 256 256

        channel_prompt_onehot = torch.softmax(self.channel_prompt / tau, dim=0)
        f_content = x * channel_prompt_onehot[0].view(1, *channel_prompt_onehot[0].shape) #[8,64,256,256]
        f_style = x * channel_prompt_onehot[1].view(1, *channel_prompt_onehot[1].shape)

                ###
        # self.sfs[0].features.shape
        # torch.Size([8, 64, 256, 256])
        # self.sfs[1].features.shape
        # torch.Size([8, 64, 128, 128])
        # self.sfs[2].features.shape
        # torch.Size([8, 128, 64, 64])
        # self.sfs[3].features.shape
        # torch.Size([8, 256, 32, 32])
        # len (self.sfs)
        # 4

        x = F.relu(self.rn(f_content)) #[8,512,16,16] 
        proj=x
        x = self.up1(x, self.sfs[3].features) #[8,256,32,32]
        x = self.up2(x, self.sfs[2].features)#[8,256,64,64]
        x = self.up3(x, self.sfs[1].features)#[8,256,128,128]
        x = self.up4(x, self.sfs[0].features)#[8,256,256,256]
        output = self.up5(x) #[8,2,512,512]

        return proj, output 

    def close(self):
        for sf in self.sfs: sf.remove()
