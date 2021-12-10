import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import util

class ASLImagenetNet(nn.Module):
    def __init__(self):
        super(ASLImagenetNet, self).__init__()
        # TODO define the layers
        self.conv0 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.conv0_bn = nn.BatchNorm2d(64)
        self.pool0 = nn.MaxPool2d(2, 2)
        self.dropout0 = nn.Dropout(0.25)
        self.conv10 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv10_bn = nn.BatchNorm2d(128)
        self.conv11 = nn.Conv2d(128, 128, 1, stride=1)
        self.conv11_bn = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(64, 128, 1, stride=2)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.conv20 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv20_bn = nn.BatchNorm2d(256)
        self.conv21 = nn.Conv2d(256, 256, 1, stride=1)
        self.conv21_bn = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(128, 256, 1, stride=2)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv30 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.conv30_bn = nn.BatchNorm2d(512)
        self.conv31 = nn.Conv2d(512, 512, 1, stride=1)
        self.conv31_bn = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(256, 512, 1, stride=2)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.conv40 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)
        self.conv40_bn = nn.BatchNorm2d(1024)
        self.conv41 = nn.Conv2d(1024, 1024, 1, stride=1)
        self.conv41_bn = nn.BatchNorm2d(1024)
        self.conv4 = nn.Conv2d(512, 1024, 1, stride=2)
        self.conv4_bn = nn.BatchNorm2d(1024)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(1024, 200)
        self.accuracy = None

    def forward(self, x):
        # TODO define the forward pass
        x = self.conv0_bn(self.conv0(x))
        x = F.leaky_relu(x)
        x = self.dropout0(self.pool0(x))
        # print(x.shape)

        x1 = self.conv10_bn(self.conv10(x))
        x1 = F.leaky_relu(x1)
        x1 = self.conv11_bn(self.conv11(x1))
        x1 = self.conv1_bn(self.conv1(x)) + x1
        x1 = F.leaky_relu(x1)
        # print(x1.shape)
        
        x2 = self.conv20_bn(self.conv20(x1))
        x2 = F.leaky_relu(x2)
        x2 = self.conv21_bn(self.conv21(x2))
        x2 = self.conv2_bn(self.conv2(x1)) + x2
        x2 = F.leaky_relu(x2)
        # print(x2.shape)

        x3 = self.conv30_bn(self.conv30(x2))
        x3 = F.leaky_relu(x3)
        x3 = self.conv31_bn(self.conv31(x3))
        x3 = self.conv3_bn(self.conv3(x2)) + x3
        x3 = F.leaky_relu(x3)
        # print(x3.shape)

        x4 = self.conv40_bn(self.conv40(x3))
        x4 = F.leaky_relu(x4)
        x4 = self.conv41_bn(self.conv41(x4))
        x4 = self.conv4_bn(self.conv4(x3)) + x4
        x4 = F.leaky_relu(x4)
        # print(x4.shape)

        x4 = self.dropout1(self.pool1(x4))
        # print(x4.shape)
        x4 = torch.flatten(x4, 1)
        # print(x4.shape)
        x4 = self.fc1(x4)
        # print(x4.shape)

        return x4


class Darknet64(nn.Module):
    def __init__(self, loss=nn.CrossEntropyLoss()):
        super(Darknet64, self).__init__()
        self.loss = loss
        self.accuracy = None
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256, 1000)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), kernel_size=2, stride=2) # 32x32x16
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), kernel_size=2, stride=2) # 16x16x32
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), kernel_size=2, stride=2) # 8x8x64
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), kernel_size=2, stride=2) # 4x4x128
        x = F.max_pool2d(F.relu(self.bn5(self.conv5(x))), kernel_size=2, stride=2) # 2x2x256

        # Global average pooling across each channel (Input could be 2x2x256, 4x4x256, 7x3x256, output would always be 256 length vector)
        x = F.adaptive_avg_pool2d(x, 1)                                            # 1x1x256
        x = torch.flatten(x, 1)                                                    # vector 256
        
        
        x = self.fc1(x)
        return x


def Resnet():
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Linear(512, 26)
    resnet.accuracy = None
    return resnet

def Resnext():
    resnext = models.resnext50_32x4d(pretrained=True)
    resnext.fc = nn.Linear(2048, 26)
    resnext.accuracy = None
    return resnext

