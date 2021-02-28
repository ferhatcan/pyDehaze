import torch
import torch.nn as nn
from torchvision import models

from models.IModel import IModel

"""
:source https://github.com/usuyama/pytorch-unet
"""

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class EncoderResNet(IModel):
    def __init__(self, args):
        super(EncoderResNet, self).__init__(args)

        self.base_model = models.resnet50(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(256, 256, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(512, 512, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(1024, 1024, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(2048, 2048, 1, 0)

    def forward(self, x):
        out = dict()
        x_original = self.conv_original_size0(x)
        x_original = self.conv_original_size1(x_original)
        out['original'] = x_original

        out['layer0'] = self.layer0(x)
        out['layer1'] = self.layer1(out['layer0'])
        out['layer2'] = self.layer2(out['layer1'])
        out['layer3'] = self.layer3(out['layer2'])
        out['layer4'] = self.layer4(out['layer3'])

        out['layer0'] = self.layer0_1x1(out['layer0'])
        out['layer1'] = self.layer1_1x1(out['layer1'])
        out['layer2'] = self.layer2_1x1(out['layer2'])
        out['layer3'] = self.layer3_1x1(out['layer3'])
        out['layer4'] = self.layer4_1x1(out['layer4'])

        return out

class DecoderResNet(IModel):
    def __init__(self, args):
        super(DecoderResNet, self).__init__(args)

        self.output_dim = args.output_dim if hasattr(args, 'output_dim') else 3

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(2048 + 1024, 1024, 3, 1)
        self.conv_up2 = convrelu(512 + 1024, 512, 3, 1)
        self.conv_up1 = convrelu(256 + 512, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size = convrelu(64 + 128, 64, 3, 1)
        self.conv_last = nn.Conv2d(64, self.output_dim, 1)


    def forward(self, x):
        out = self.upsample(x['layer4'])
        out = torch.cat([out, x['layer3']], dim=1)
        out = self.conv_up3(out)

        out = self.upsample(out)
        out = torch.cat([out, x['layer2']], dim=1)
        out = self.conv_up2(out)

        out = self.upsample(out)
        out = torch.cat([out, x['layer1']], dim=1)
        out = self.conv_up1(out)

        out = self.upsample(out)
        out = torch.cat([out, x['layer0']], dim=1)
        out = self.conv_up0(out)

        out = self.upsample(out)
        out = torch.cat([out, x['original']], dim=1)
        out = self.conv_original_size(out)

        out = self.conv_last(out)

        return out


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = ResNetUNet(n_class=3)
# model = model.to(device)
#
# # check keras-like model summary using torchsummary
# from torchsummary import summary
# summary(model, input_size=(3, 256, 256))
#
# # resnet50 = models.resnet50(pretrained=True).to(device)
# # summary(resnet50, input_size=(3, 256, 256))
#
# encoderDecoderResnet = EncoderDecoderResnet(None).to(device)
# inp = torch.randn((1, 3, 256, 256)).to(device)
# out = encoderDecoderResnet(inp)
#
# out2 = model(inp)
# tmp = 0
