import torch.nn.init as init

from ddp.ddp import DDP, DDP_input
from models.ffc import ConcatTupleLayer, FFCResnetBlock

from .unet_parts import *


class Falcon(nn.Module):
    def __init__(self, n_channels, out_channels=3, input_kernel=[5], config_ffc=None, bilinear=False):
        super(Falcon, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.config_ffc = config_ffc
        self.input_kernel = input_kernel
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        
        self.middle_part = FFCResnetBlock(512, padding_type='reflect', activation_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, config_ffc=config_ffc)
        self.middle_concat = ConcatTupleLayer()
        
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, out_channels))
        self.ddp = DDP(kernel_size=15)
        self.ddp_input = DDP_input()
        
    def _init_weights(self, m, name):
        if name == 'he_u':
            init_fn = init.kaiming_uniform_
        if name == 'he_n':
            init_fn = init.kaiming_normal_
        if name == 'xavier':
            init_fn = init.xavier_uniform_
        
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            init_fn(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init_fn(m.weight)
            init.constant_(m.bias, 0)
        
    def _initialize_weights(self, name):
        for m in self.modules():
            self._init_weights(m, name)
        print(f"Model is initialized with {name}")
        
    def forward(self, x, w=1, test=False):
        x_ = []
        for b in range(x.size(0)):
            for k in self.input_kernel:
                x_.append(self.ddp_input(x[b:b+1,...], k))
        x_ = torch.cat(x_).view(x.size(0), -1, x.size(2), x.size(3)).contiguous()
        x = torch.cat((x, x_), dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        for i in range(self.config_ffc['loop']):
            x4 = self.middle_part(x4)
        x4 = self.middle_concat(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if test:
            return logits
        t_haze = w * self.ddp(logits)
        return logits, t_haze

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)