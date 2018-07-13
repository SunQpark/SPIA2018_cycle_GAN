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

    def __init__(self, in_ch=3, out_ch=3, ch=16):
        super(ResNet6, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 7, stride=1, padding=3, bias=False)
        self.inorm1 = nn.InstanceNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=0, bias=False)
        self.inorm2 = nn.InstanceNorm2d(64)
        # self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=0, bias=False)
        # self.inorm3 = nn.InstanceNorm2d(128)

        self.block1 = BasicBlock(64, 64)
        self.block2 = BasicBlock(64, 64)
        self.block3 = BasicBlock(64, 64)
        self.block4 = BasicBlock(64, 64)
        self.block5 = BasicBlock(64, 64)
        self.block6 = BasicBlock(64, 64)

        # self.conv4_t = nn.ConvTranspose2d(128, 64, 3, 2, padding=0, bias=False)
        # self.inorm4 = nn.InstanceNorm2d(64)
        self.conv5_t = nn.ConvTranspose2d(64, 32, 4, 2, padding=0, bias=False)
        self.inorm5 = nn.InstanceNorm2d(32)
        self.conv6 = nn.Conv2d(32, out_ch, 7, 1, padding=3, bias=False)
        self.inorm6 = nn.InstanceNorm2d(out_ch)


    def forward(self, x_input):
        x = F.relu(self.inorm1(self.conv1(x_input)))
        x = F.relu(self.inorm2(self.conv2(x)))
        # x = F.relu(self.inorm3(self.conv3(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        # x = F.relu(self.inorm4(self.conv4_t(x)))
        x = F.relu(self.inorm5(self.conv5_t(x)))
        x = F.tanh(self.inorm6(self.conv6(x)))
        return x

class ResNet9(BaseModel):

    def __init__(self, in_ch=3, out_ch=3, ch=16):
        super(ResNet9, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 7, stride=1, padding=3, bias=False)
        self.inorm1 = nn.InstanceNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=0, bias=False)
        self.inorm2 = nn.InstanceNorm2d(64)
        # self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=0, bias=False)
        # self.inorm3 = nn.InstanceNorm2d(128)

        self.block1 = BasicBlock(64, 64)
        self.block2 = BasicBlock(64, 64)
        self.block3 = BasicBlock(64, 64)
        self.block4 = BasicBlock(64, 64)
        self.block5 = BasicBlock(64, 64)
        self.block6 = BasicBlock(64, 64)
        self.block7 = BasicBlock(64, 64)
        self.block8 = BasicBlock(64, 64)
        self.block9 = BasicBlock(64, 64)

        # self.conv4_t = nn.ConvTranspose2d(128, 64, 3, 2, padding=0, bias=False)
        # self.inorm4 = nn.InstanceNorm2d(64)
        self.conv5_t = nn.ConvTranspose2d(64, 32, 4, 2, padding=0, bias=False)
        self.inorm5 = nn.InstanceNorm2d(32)
        self.conv6 = nn.Conv2d(32, out_ch, 7, 1, padding=3, bias=False)
        self.inorm6 = nn.InstanceNorm2d(out_ch)


    def forward(self, x_input):
        x = F.relu(self.inorm1(self.conv1(x_input)))
        x = F.relu(self.inorm2(self.conv2(x)))
        # x = F.relu(self.inorm3(self.conv3(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        # x = F.relu(self.inorm4(self.conv4_t(x)))
        x = F.relu(self.inorm5(self.conv5_t(x)))
        x = F.tanh(self.inorm6(self.conv6(x)))
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

        self.conv4 = nn.Conv2d(256, 512, 3, 1, padding=1, bias=False)
        self.inorm4 = nn.InstanceNorm2d(512)

        self.conv5 = nn.Conv2d(512,   1, 1, 1, padding=0, bias=False)
        
        
    def forward(self, x):
        x = F.leaky_relu_(self.conv1(x), 0.2)
        x = F.leaky_relu_(self.inorm2(self.conv2(x)), 0.2)
        x = F.leaky_relu_(self.inorm3(self.conv3(x)), 0.2)
        x = F.leaky_relu_(self.inorm4(self.conv4(x)), 0.2)
        return F.sigmoid(self.conv5(x))

class GAN_Res6(BaseModel):
    def __init__(self, in_ch=3, out_ch=3, ch=16):
        super(GAN_Res6, self).__init__()
        self.gen = ResNet6(in_ch=in_ch, out_ch=out_ch, ch=ch)
        self.dis = Discriminator(in_ch=out_ch)
        

    def forward(self, x_input, discrim=True):
        fake_y = self.gen(x_input)
        return fake_y

class GAN_Res9(BaseModel):
    def __init__(self, in_ch=3, out_ch=3, ch=16):
        super(GAN_Res9, self).__init__()
        self.gen = ResNet9(in_ch=in_ch, out_ch=out_ch, ch=ch)
        self.dis = Discriminator(in_ch=out_ch)
        
    def forward(self, x_input, discrim=True):
        fake_y = self.gen(x_input)
        return fake_y


class CycleGAN(BaseModel):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.AtoB = GAN_Res9(in_ch=1, ch=64)
        self.BtoA = GAN_Res6(out_ch=1, ch=32)

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
    gen = ResNet6()
    dis = Discriminator()

    x_dummy = torch.randn((1, 3, 256, 256))
    x_inter = gen(x_dummy)
    print(x_inter.shape)
    output = dis(x_inter)
    print(output.shape)