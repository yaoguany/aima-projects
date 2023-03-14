import torch
import torch.nn as nn
import torch.nn.functional as F

#定义残差块ResBlock

#每两个卷积层之间有一个残差链接
#故每两个卷积层定义为一个block
#每两个block为一个layer
class Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample = None, stride = 1):
        super(Block, self).__init__()
        #Convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        #batch normalization
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        #将ReLU激活函数改为了抑制区0.2的LeakyReLU激活函数
        self.relu = nn.LeakyReLU(inplace = True, negative_slope = 0.2)
        #残差块
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class ResNet_18(nn.Module):
    
    def __init__(self, image_channels = 3, num_classes = 8):
        
        super(ResNet_18, self).__init__()
        #通过一个7*7*64的卷积，步长设置为2，使得图像的大小缩小了一半
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace = True, negative_slope = 0.2)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        #resnet layers
        #对于layer1，把stride设置为1，使其通过一个最大值池化
        self.layer1 = self.__make_layer(64, 64, stride = 1)
        self.layer2 = self.__make_layer(64, 128, stride = 2)
        self.layer3 = self.__make_layer(128, 256, stride = 2)
        self.layer4 = self.__make_layer(256, 512, stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def __make_layer(self, in_channels, out_channels, stride):
        
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
            
        return nn.Sequential(
            #一个layer为两个block
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
            Block(out_channels, out_channels)
        )
        
    def forward(self, x):
        #初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        #四个layer
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #自适应平均池化层
        x = self.avgpool(x)
        #展开为一维
        x = x.view(x.shape[0], -1)
        #全连接层
        x = self.fc(x)
        #Softmax概率归一化
        x = nn.Softmax(dim = -1)(x)
        return x 
    
    #下载样例
    def identity_downsample(self, in_channels, out_channels):
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1), 
            nn.BatchNorm2d(out_channels)
        )