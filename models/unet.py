import torch
import torch.nn as nn

class Convnd(nn.Module):
    def __init__(self, num_dims, in_channels, out_channels, stride) -> None:
        super().__init__()
        convnd = nn.Conv3d if num_dims == 3 else nn.Conv2d
        normnd = nn.InstanceNorm3d if num_dims == 3 else nn.InstanceNorm2d
        self.conv = convnd(in_channels, out_channels, 3, stride, 1)
        self.norm = normnd(out_channels, affine=True)
        self.relu = nn.LeakyReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        return self.relu(self.norm(x))

class StackedConvnd(nn.Module):
    def __init__(self, num_dims, in_channels, out_channels, first_stride) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            Convnd(num_dims, in_channels, out_channels, first_stride),
            Convnd(num_dims, out_channels, out_channels, 1)
        )
    
    def forward(self, x):
        return self.blocks(x)


class UNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_dims = config.MODEL.NUM_DIMS
        assert self.num_dims in [2, 3], 'only 2d or 3d inputs are supported'
        self.num_classes = config.DATASET.NUM_CLASSES

        self.extra = config.MODEL.EXTRA
        self.enc_channels = self.extra.ENC_CHANNELS
        self.dec_channels = self.extra.DEC_CHANNELS
        
        # encoder
        self.enc = nn.ModuleList()
        prev_channels = config.DATASET.NUM_MODALS
        for i, channels in enumerate(self.enc_channels):
            # we do not perform downsampling at first convolution layer
            first_stride = 2 if i != 0 else 1
            self.enc.append(StackedConvnd(self.num_dims, prev_channels, channels, first_stride))
            prev_channels = channels

        # decoder
        self.dec_up = nn.ModuleList()
        self.dec_cat = nn.ModuleList()
        prev_channels = channels
        deconvnd = nn.ConvTranspose3d if self.num_dims == 3 else nn.ConvTranspose2d
        for channels in self.dec_channels:
            self.dec_up.append(deconvnd(prev_channels, channels, 2, 2, bias=False))
            self.dec_cat.append(
                nn.Sequential(
                    StackedConvnd(self.num_dims, channels*2, channels, 1),  # Concat skip features
                    StackedConvnd(self.num_dims, channels, channels, 1)
                )
            )
            prev_channels = channels

        # outputs
        convnd = nn.Conv3d if self.num_dims == 3 else nn.Conv2d
        self.conv_segs = nn.ModuleList()
        for channels in self.dec_channels[1:]:
            self.conv_segs.append(convnd(channels, self.num_classes, 1, 1, 0, bias=False))

    def forward(self, x):
        skips, outs = [], []
        
        for layer in self.enc[:-1]:
            x = layer(x)
            skips.append(x)
        x = self.enc[-1](x)
        for i, (layer_up, layer_cat) in enumerate(zip(self.dec_up, self.dec_cat)):
            x = layer_up(x)
            x = torch.cat([x, skips[-(i+1)]], dim=1)
            x = layer_cat(x)
            if i >= 1:
                outs.append(self.conv_segs[i-1](x))
        outs.reverse()
        return outs
