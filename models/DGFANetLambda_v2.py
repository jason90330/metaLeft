import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
from collections import OrderedDict
from torch.autograd import Variable
import random
from pdb import set_trace as st



def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias)



class inconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(inconv, self).__init__()

        self.conv = nn.Sequential(
                    conv3x3(in_channels, out_channels),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x    


class Downconv(nn.Module):
    """
    A helper Module that performs 3 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels):
        super(Downconv, self).__init__()

        self.downconv = nn.Sequential(
            conv3x3(in_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            conv3x3(128, 196),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),

            conv3x3(196, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.downconv(x)
        return x


class DepthEstmator(nn.Module):
    def __init__(self, in_channels=832, out_channels=1):
    # def __init__(self, in_channels=384, out_channels=1):
        super(DepthEstmator, self).__init__()

        self.conv = nn.Sequential(
            conv3x3(in_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            conv3x3(128, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),            
            
            conv3x3(64, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) 
        )


    def forward(self, x):
        x = self.conv(x)
        return x

class LambdaConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, k=16, u=1, m=23):
        super(LambdaConv, self).__init__()
        self.kk, self.uu, self.vv, self.mm, self.heads = k, u, out_channels // heads, m, heads
        self.local_context = True if m > 0 else False
        self.padding = (m - 1) // 2

        self.queries = nn.Sequential(
            nn.Conv2d(in_channels, k * heads, kernel_size=1, bias=False),
            nn.BatchNorm2d(k * heads)
        )
        self.keys = nn.Sequential(
            nn.Conv2d(in_channels, k * u, kernel_size=1, bias=False),
        )
        self.values = nn.Sequential(
            nn.Conv2d(in_channels, self.vv * u, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.vv * u)
        )

        self.softmax = nn.Softmax(dim=-1)

        if self.local_context:
            self.embedding = nn.Parameter(torch.randn([self.kk, self.uu, 1, m, m]), requires_grad=True)
        else:
            self.embedding = nn.Parameter(torch.randn([self.kk, self.uu]), requires_grad=True)

    def forward(self, x):
        n_batch, C, w, h = x.size()

        queries = self.queries(x).view(n_batch, self.heads, self.kk, w * h) # b, heads, k // heads, w * h
        softmax = self.softmax(self.keys(x).view(n_batch, self.kk, self.uu, w * h)) # b, k, uu, w * h
        values = self.values(x).view(n_batch, self.vv, self.uu, w * h) # b, v, uu, w * h

        lambda_c = torch.einsum('bkum,bvum->bkv', softmax, values)
        y_c = torch.einsum('bhkn,bkv->bhvn', queries, lambda_c)

        if self.local_context:
            values = values.view(n_batch, self.uu, -1, w, h)
            lambda_p = F.conv3d(values, self.embedding, padding=(0, self.padding, self.padding))
            lambda_p = lambda_p.view(n_batch, self.kk, self.vv, w * h)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)
        else:
            lambda_p = torch.einsum('ku,bvun->bkvn', self.embedding, values)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)

        out = y_c + y_p
        out = out.contiguous().view(n_batch, -1, w, h)

        return out

class LambdaBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(LambdaBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.ModuleList([LambdaConv(planes, planes)])
        if stride != 1 or in_planes != self.expansion * planes:
            self.conv2.append(nn.AvgPool2d(kernel_size=(3, 3), stride=stride, padding=(1, 1)))
        self.conv2.append(nn.BatchNorm2d(planes))
        self.conv2.append(nn.ReLU())
        self.conv2 = nn.Sequential(*self.conv2)

        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ImageNet 350 epochs training setup
        # self.maxpool = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )

        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Sequential(
        #     nn.Dropout(0.3), # All architecture deeper than ResNet-200 dropout_rate: 0.2
        #     nn.Linear(512 * block.expansion, num_classes)
        # )

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):#6*6*256*256
        dx1 = self.relu(self.bn1(self.conv1(x)))#6*64*128*128
        dx1_mp = self.maxpool(dx1)#6*64*64*64
        dx2 = self.layer1(dx1_mp)#6*256*64*64
        dx3 = self.layer2(dx2)#6*512*32*32
        # dx4 = self.layer3(dx3)#6*1024*16*16
        # dx5 = self.layer4(dx4)#6*2048*8*8
        '''
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
        '''
        return dx1_mp, dx2, dx3

class ResNet_back(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet_back, self).__init__()
        self.in_planes = 512

        # self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ImageNet 350 epochs training setup
        # self.maxpool = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )

        # self.layer1 = self._make_layer(block, 64, num_blocks[0])
        # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.3), # All architecture deeper than ResNet-200 dropout_rate: 0.2
            nn.Linear(512 * block.expansion, num_classes)
        )

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):#30*512*32*32
        # dx1 = self.relu(self.bn1(self.conv1(x)))#6*64*128*128
        # dx1_mp = self.maxpool(dx1)#6*64*64*64
        # dx2 = self.layer1(dx1_mp)#6*256*64*64
        # dx3 = self.layer2(dx2)#6*512*32*32
        out = self.layer3(x)#30*1024*16*16
        out = self.layer4(out)#30*2048*8*8
        out = self.avgpool(out)#30*2048*1*1
        out = torch.flatten(out, 1)#30*2048
        out = self.fc(out)#30*1
        return out
        '''
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
        '''
        return out
    
def LambdaResNet50():
    return ResNet(LambdaBottleneck, [3, 4, 6, 3],num_classes=1)

def LambdaResNet50_back():
    return ResNet_back(LambdaBottleneck, [3, 4, 6, 3],num_classes=1)

class FeatExtractor(nn.Module):
    def __init__(self, in_channels=6): # rgb+hsv
        super(FeatExtractor, self).__init__()  
        self.res = LambdaResNet50()
        '''
        self.inc = inconv(in_channels, 64)

        self.down1 = Downconv(64, 128)
        self.down2 = Downconv(128, 128)
        self.down3 = Downconv(128, 128)
        '''
    def forward(self, x):
        dx1_mp, dx2, dx3 = self.res(x)#6*64*64*64, 6*256*64*64, 6*512*32*32
        re_dx1 = F.adaptive_avg_pool2d(dx1_mp, 32)#6*512*32*32
        re_dx2 = F.adaptive_avg_pool2d(dx2, 32)#6*1024*32*32
        catfeat = torch.cat([re_dx1, re_dx2, dx3],1)
        '''
        dx1 = self.inc(x)
        dx2 = self.down1(dx1)
        dx3 = self.down2(dx2)
        dx4 = self.down3(dx3)

        re_dx2 = F.adaptive_avg_pool2d(dx2, 32)
        re_dx3 = F.adaptive_avg_pool2d(dx3, 32)
        catfeat = torch.cat([re_dx2, re_dx3, dx4],1)
        '''
        return catfeat, dx3

class FeatEmbedder(nn.Module):
    def __init__(self, in_channels=512,momentum=0.1):
    # def __init__(self, in_channels=128,momentum=0.1):
        super(FeatEmbedder, self).__init__()  
        self.resBack = LambdaResNet50_back()
        
        '''
        self.momentum = momentum

        self.features = nn.Sequential(
            conv_block(0, in_channels=in_channels, out_channels=128, momentum=self.momentum, pooling=True),
            conv_block(1, in_channels=128, out_channels=256, momentum=self.momentum,pooling=True),
            conv_block(2, in_channels=256, out_channels=512, momentum=self.momentum,pooling=False),
            nn.AdaptiveAvgPool2d((1, 1))
            )        
        self.add_module('fc', nn.Linear(512, 1))
        '''
    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, params=None):
        out = self.resBack(x)
        return out
        '''
        if params == None:
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
        else:

            out = F.conv2d(
                x,
                params['features.0.conv0.weight'],
                params['features.0.conv0.bias'],
                padding=1)
            out = F.batch_norm(
                out,
                params['features.0.bn0.running_mean'],
                params['features.0.bn0.running_var'],
                params['features.0.bn0.weight'],
                params['features.0.bn0.bias'],
                momentum=self.momentum,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, 2)


            out = F.conv2d(
                out,
                params['features.1.conv1.weight'],
                params['features.1.conv1.bias'],
                padding=1)
            out = F.batch_norm(
                out,
                params['features.1.bn1.running_mean'],
                params['features.1.bn1.running_var'],
                params['features.1.bn1.weight'],
                params['features.1.bn1.bias'],
                momentum=self.momentum,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, 2)

            out = F.conv2d(
                out,
                params['features.2.conv2.weight'],
                params['features.2.conv2.bias'],
                padding=1)
            out = F.batch_norm(
                out,
                params['features.2.bn2.running_mean'],
                params['features.2.bn2.running_var'],
                params['features.2.bn2.weight'],
                params['features.2.bn2.bias'],
                momentum=self.momentum,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.adaptive_avg_pool2d(out,1)

            out = out.view(out.size(0), -1)
            out = F.linear(out, params['fc.weight'],
                           params['fc.bias'])        
        return out
        '''

    def cloned_state_dict(self):
        cloned_state_dict = {
            key: val.clone()
            for key, val in self.state_dict().items()
        }
        return cloned_state_dict    


def conv_block(index,
               in_channels,
               out_channels,
               K_SIZE=3,
               stride=1,
               padding=1,
               momentum=0.1,
               pooling=True):
    """
    The unit architecture (Convolutional Block; CB) used in the modules.
    The CB consists of following modules in the order:
        3x3 conv, 64 filters
        batch normalization
        ReLU
        MaxPool
    """
    if pooling:
        conv = nn.Sequential(
            OrderedDict([
                ('conv'+str(index), nn.Conv2d(in_channels, out_channels, \
                    K_SIZE, stride=stride, padding=padding)),
                ('bn'+str(index), nn.BatchNorm2d(out_channels, momentum=momentum, \
                    affine=True)),
                ('relu'+str(index), nn.ReLU(inplace=True)),
                ('pool'+str(index), nn.MaxPool2d(2))
            ]))
    else:
        conv = nn.Sequential(
            OrderedDict([
                ('conv'+str(index), nn.Conv2d(in_channels, out_channels, \
                    K_SIZE, padding=padding)),
                ('bn'+str(index), nn.BatchNorm2d(out_channels, momentum=momentum, \
                    affine=True)),
                ('relu'+str(index), nn.ReLU(inplace=True))
            ]))
    return conv



