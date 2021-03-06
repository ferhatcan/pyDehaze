import torch
import torch.nn as nn

from models.IModel import IModel


class Encoder(IModel):
    def __init__(self, args):
        super(Encoder, self).__init__(args)

        self.input_dim = args.input_dim if hasattr(args, 'input_dim') else 3
        self.maxpool = nn.AvgPool2d(2)

        self.encoder_layer_0 = nn.Conv2d(self.input_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder_layer_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder_layer_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder_layer_3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder_layer_4 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        output = dict()
        output['input_image'] = x
        output['input_layer'] = self.maxpool(self.encoder_layer_0(x))
        output['layer1'] = self.maxpool(self.encoder_layer_1(output['input_layer']))
        output['layer2'] = self.maxpool(self.encoder_layer_2(output['layer1']))
        output['layer3'] = self.maxpool(self.encoder_layer_3(output['layer2']))
        output['layer4'] = self.maxpool(self.encoder_layer_4(output['layer3']))

        return output


class Decoder(IModel):
    def __init__(self, args):
        super(Decoder, self).__init__(args)

        self.args = args

        self.output_dim = args.output_dim if hasattr(args, 'output_dim') else 3
        self.include_input_image = args.include_input_image if hasattr(args, 'include_input_image') else False


        self.layer1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=(1, 1), groups=32, bias=False)
        self.layer2 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=(1, 1), groups=32, bias=False)
        self.layer3 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=(1, 1), groups=32, bias=False)
        self.layer4 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=(1, 1), groups=32, bias=False)
        self.layer5 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=(1, 1), groups=32, bias=False)
        self.layer6 = nn.Conv2d(64, self.output_dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.tanh = nn.Tanh()

        self.output_layer = nn.Sequential(
            self.layer6,
            self.tanh,
        )

    def forward(self, feats):
        assert len(feats) >= 5, 'There should be minimum 5 layer outputs for decoder!'

        out = self.layer1(feats['layer4'])
        out = torch.cat((feats['layer3'], out), dim=1)

        out = self.layer2(out)
        out = torch.cat((feats['layer2'], out), dim=1)

        out = self.layer3(out)
        out = torch.cat((feats['layer1'], out), dim=1)

        out = self.layer4(out)
        out = torch.cat((feats['input_layer'], out), dim=1)

        out = self.layer5(out)

        out = self.output_layer(out)

        if self.include_input_image and 'input_image' in feats:
            out = (out + feats['input_image']) / 2

        return out


class DecoderResizeConv(Decoder):
    def __init__(self, args):
        super(DecoderResizeConv, self).__init__(args)

        self.layer1 = ResizeConv2D(1024, 512)
        self.layer2 = ResizeConv2D(1024, 256)
        self.layer3 = ResizeConv2D(512, 128)
        self.layer4 = ResizeConv2D(256, 64)
        self.layer5 = ResizeConv2D(128, 64)


class ResizeConv2D(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 bias = False,
                 scale_factor=2,
                 mode='bilinear'):
        super(ResizeConv2D, self).__init__()

        self.resize = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        out = self.resize(x)
        out = self.conv(out)

        return out

# Test starts

# decoder = DecoderResizeConv(None)
#
# output = dict()
# output['input_layer'] = torch.randn(1, 64, 32, 32)
# output['layer1'] = torch.randn(1, 128, 16, 16)
# output['layer2'] = torch.randn(1, 256, 8, 8)
# output['layer3'] = torch.randn(1, 512, 4, 4)
# output['layer4'] = torch.randn(1, 1024, 2, 2)
#
# out = decoder(output)
#
# tmp = 0