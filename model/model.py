import sys 
sys.path.append('./')
from base import BaseModel
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, in_ch, ch, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.downsample = downsample

    def forward(self, x_input):
        residual = x_input
        x = self.relu(self.bn1(self.conv1(x_input)))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            residual = self.downsample(residual)
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


class ResNet6(BaseModel):

    def __init__(self, ):
        super(ResNet6, self).__init__()
        self.conv = nn.Conv2d(3,   16, 4, 2, padding=1, bias=False)
        self.block1 = BasicBlock(16, 16)
        self.block2 = BasicBlock(16, 16)
        self.block3 = BasicBlock(16, 16)
        self.block4 = BasicBlock(16, 16)
        self.block5 = BasicBlock(16, 16)
        self.block6 = BasicBlock(16, 16)
        self.conv_t = nn.ConvTranspose2d(16, 3, 4, 2, padding=1, bias=False)

    def forward(self, x_input):
        x = self.conv(x_input)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.conv_t(x)
        return x


class Discriminator(BaseModel):

    def __init__(self, n_c=3, n_size=4):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3,   64, 4, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 4, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, 3, 1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512,   1, 3, 1, padding=0, bias=False)
        
        
    def forward(self, x):
        x = F.leaky_relu_(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu_(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu_(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu_(self.bn4(self.conv4(x)), 0.2)
        return F.tanh(self.conv5(x))

class GAN(BaseModel):
    def __init__(self):
        super(GAN, self).__init__()
        self.gen = ResNet6()
        self.dis = Discriminator()
        

    def forward(self, x_input, discrim=True):
        y_fake = self.gen(x_input)
        if discrim:
            score = self.dis(y_fake)
            return score, y_fake
        else:
            return y_fake


class CycleGAN(BaseModel):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.AtoB = GAN()
        self.BtoA = GAN()

    def forward(x, from_A=True):
        if from_A:
            fake_y = self.AtoB.gen(x)
            recon_x = self.BtoA.gen(fake_y)
            if self.training:
                score = self.AtoB.dis(fake_y)
                return fake_y, recon_x, score
            else:
                return fake_y, recon_x
        else:
            fake_y = self.BtoA.gen(x)
            recon_x = self.AtoB.gen(fake_y)
            if self.training:
                score = self.BtoA.dis(fake_y)
                return fake_y, recon_x, score
            else:
                return fake_y, recon_x



if __name__ == '__main__':
    import torch
    gen = ResNet6()
    dis = Discriminator()

    x_dummy = torch.randn((1, 3, 256, 256))
    x_inter = gen(x_dummy)
    print(x_inter.shape)
    output = dis(x_inter)
    print(output.shape)