import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class skipconnection(nn.Module):
    def __init__(self, skip):
        super(skipconnection, self).__init__()
        self.skip = skip
    
    def forward(self, x):
        return self.skip(x)

class BasicBlock(nn.Module):
    def __init__(self, in_dim, dim, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

        self.shortcut = nn.Sequential()
        if stride!=1 or in_dim != dim:
            self.shortcut=skipconnection(lambda x:
                                        F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, dim//4, dim//4), "constant", 0))
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, n, r=10):
        super(ResNet, self).__init__()
        self.in_dim = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, n, stride=1)
        self.layer2 = self._make_layer(block, 64, n, stride=2)
        self.layer3 = self._make_layer(block, 128, n, stride=2)
        self.linear = nn.Linear(128, r)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.apply(_weights_init)

    def _make_layer(self, block, dim, n, stride):
        strides = [stride] + [1]*(n-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_dim, dim, stride))
            self.in_dim = dim
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

def resnet(n, r):
    return ResNet(BasicBlock, n, r)