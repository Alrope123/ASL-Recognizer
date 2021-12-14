import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ASLImagenetNet(nn.Module):
    def __init__(self):
        super(ASLImagenetNet, self).__init__()
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
        x = self.conv0_bn(self.conv0(x))
        x = F.leaky_relu(x)
        x = self.dropout0(self.pool0(x))

        x1 = self.conv10_bn(self.conv10(x))
        x1 = F.leaky_relu(x1)
        x1 = self.conv11_bn(self.conv11(x1))
        x1 = self.conv1_bn(self.conv1(x)) + x1
        x1 = F.leaky_relu(x1)
        
        x2 = self.conv20_bn(self.conv20(x1))
        x2 = F.leaky_relu(x2)
        x2 = self.conv21_bn(self.conv21(x2))
        x2 = self.conv2_bn(self.conv2(x1)) + x2
        x2 = F.leaky_relu(x2)

        x3 = self.conv30_bn(self.conv30(x2))
        x3 = F.leaky_relu(x3)
        x3 = self.conv31_bn(self.conv31(x3))
        x3 = self.conv3_bn(self.conv3(x2)) + x3
        x3 = F.leaky_relu(x3)

        x4 = self.conv40_bn(self.conv40(x3))
        x4 = F.leaky_relu(x4)
        x4 = self.conv41_bn(self.conv41(x4))
        x4 = self.conv4_bn(self.conv4(x3)) + x4
        x4 = F.leaky_relu(x4)

        x4 = self.dropout1(self.pool1(x4))
        x4 = torch.flatten(x4, 1)
        x4 = self.fc1(x4)

        return x4


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

