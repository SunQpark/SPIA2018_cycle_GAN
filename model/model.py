import sys 
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('./')
from base import BaseModel



class BasicBlock(nn.Module):

    def __init__(self, in_ch, ch, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x_input):
        residual = x_input
        x = self.relu(self.bn1(self.conv1(x_input)))
        x = self.bn2(self.conv2(x))
        return x + residual


class BottleneckBlock(nn.Module):

    def __init__(self, in_ch, ch, stride=1, downsample=None):
        super(BottleneckBlock).__init__()
        self.conv1 = nn.Conv2d(in_ch, ch, 1, stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.conv3 = nn.Conv2d(ch, ch, 1, 1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(ch)
        self.downsample = downsample

    def forward(self, x_input):
        residual = x_input
        x = self.relu(self.bn1(self.conv1(x_input)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.downsample is not None:
            residual = self.downsample(residual)
        return x + residual


class ResNet(BaseModel):

    def __init__(self, num_layers, in_ch=3, out_ch=3, ch=16):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 7, stride=1, padding=3, bias=False)
        self.inorm1 = nn.InstanceNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
        self.inorm2 = nn.InstanceNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)
        self.inorm3 = nn.InstanceNorm2d(128)

        self.res_blocks = self._make_layers(BasicBlock, num_layers, 128)

        # upsample
        self.conv4 = nn.Conv2d(128, 64, 3, 1, padding=1, bias=False)
        self.inorm4 = nn.InstanceNorm2d(64)
        self.conv5 = nn.Conv2d(64, 32, 3, 1, padding=1, bias=False)
        self.inorm5 = nn.InstanceNorm2d(32)
        self.conv6 = nn.Conv2d(32, out_ch, 7, 1, padding=3, bias=False)
        self.inorm6 = nn.InstanceNorm2d(out_ch)

    def _make_layers(self, block, num_layers, num_channels):
        layers = [block(num_channels, num_channels)]*num_layers
        return nn.Sequential(*layers)

    def forward(self, x_input):
        x = F.relu(self.inorm1(self.conv1(x_input)))
        x = F.relu(self.inorm2(self.conv2(x)))
        x = F.relu(self.inorm3(self.conv3(x)))

        x = self.res_blocks(x)

        x = F.relu(self.inorm4(self.conv4(F.upsample(x, scale_factor=2))))
        x = F.relu(self.inorm5(self.conv5(F.upsample(x, scale_factor=2))))
        x = (F.tanh(self.inorm6(self.conv6(x))) + 1)/2
        return x


class Unet(nn.Module):

    def __init__(self, in_ch, out_ch, n_features=32):
        super(Unet, self).__init__()
        
        self.encoder = nn.Sequential(
            self._make_block(in_ch,        n_features),
            self._make_block(n_features  , n_features*2),
            self._make_block(n_features*2, n_features*4),
            self._make_block(n_features*4, n_features*8),
        )
        
        self.bottleneck = nn.Sequential(*[
            self._make_block(n_features*8,  n_features*16),
            self._make_block(n_features*16, n_features*16),

            # nn.ConvTranspose2d(n_features*16, n_features*8, 2, 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(n_features*16, n_features*8, 3, 1, padding=1),
        ])
        
        self.decoder = nn.Sequential(
            self._make_block(n_features*16, n_features*8, upsample=True),
            self._make_block(n_features*8, n_features*4, upsample=True),
            self._make_block(n_features*4, n_features*2, upsample=True),
            self._make_block(n_features*2, n_features  , upsample=False, activation=False),
        )
        self.output = nn.Conv2d(n_features, out_ch, 1, 1)
        

    def _make_block(self, in_ch, out_ch, upsample=False, activation=True):
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, 1, padding=1),
            nn.BatchNorm2d(out_ch),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
            layers.append(nn.Conv2d(out_ch, out_ch // 2, 3, 1, padding=1))

        return nn.Sequential(*layers)


    def forward(self, x):
        skip = []
        for down_block in self.encoder.children():
            x = down_block(x)
            skip.append(x)
            x = F.max_pool2d(x, 2)
        
        x = self.bottleneck(x)

        for up_block in self.decoder.children():
            x = torch.cat([skip.pop(), x], dim=1)
            x = up_block(x)

        x = self.output(x)
        x = (F.tanh(x) + 1)/2
        return x


class Discriminator(BaseModel):

    def __init__(self, in_ch=3, n_size=4):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_ch,   64, 4, 2, padding=1, bias=False)
        # self.inorm1 = nn.InstanceNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, padding=1, bias=False)
        self.inorm2 = nn.InstanceNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 4, 2, padding=1, bias=False)
        self.inorm3 = nn.InstanceNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, 4, 2, padding=1, bias=False)
        self.inorm4 = nn.InstanceNorm2d(512)

        self.conv5 = nn.Conv2d(512,   1, 1, 1, padding=0, bias=False)
        
        
    def forward(self, x):
        x = F.leaky_relu_(self.conv1(x), 0.2)
        x = F.leaky_relu_(self.inorm2(self.conv2(x)), 0.2)
        x = F.leaky_relu_(self.inorm3(self.conv3(x)), 0.2)
        x = F.leaky_relu_(self.inorm4(self.conv4(x)), 0.2)
        return F.sigmoid(self.conv5(x))

class GAN_ResNet(BaseModel):
    def __init__(self, in_ch=3, out_ch=3, num_res=6, ch=16):
        super(GAN_ResNet, self).__init__()
        self.gen = ResNet(num_res, in_ch=in_ch, out_ch=out_ch, ch=ch)
        self.dis = Discriminator(in_ch=out_ch)
        

    def forward(self, x_input, discrim=True):
        fake_y = self.gen(x_input)
        return fake_y


class GAN_UNet(BaseModel):
    def __init__(self, in_ch=3, out_ch=3, n_features=32):
        super(GAN_UNet, self).__init__()
        self.gen = Unet(in_ch=in_ch, out_ch=out_ch, n_features=n_features)
        self.dis = Discriminator(in_ch=out_ch)
        

    def forward(self, x_input, discrim=True):
        fake_y = self.gen(x_input)
        return fake_y


class CycleGAN(BaseModel):
    def __init__(self):
        super(CycleGAN, self).__init__()
        # self.AtoB = GAN_ResNet(in_ch=1, ch=64, num_res=9)
        # self.BtoA = GAN_ResNet(out_ch=1, ch=64, num_res=9)
        self.AtoB = GAN_UNet(in_ch=1,  n_features=32)
        self.BtoA = GAN_UNet(out_ch=1, n_features=32)

    def forward(self, x, from_A=True):
        if from_A:
            fake_y = self.AtoB.gen(x)
            recon_x = self.BtoA.gen(fake_y)
            return fake_y, recon_x
        else:
            fake_y = self.BtoA.gen(x)
            recon_x = self.AtoB.gen(fake_y)
            return fake_y, recon_x



if __name__ == '__main__':
    import torch
    # gen = ResNet()
    # dis = Discriminator()

    x_dummy = torch.randn((4, 3, 256, 256))
    # x_inter = gen(x_dummy)
    # print(x_inter.shape)
    # output = dis(x_inter)
    # print(output.shape)

    unet = Unet(3, 1, 32)
    print(unet)
    output = unet(x_dummy)
    print(output.shape)