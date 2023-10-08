import torch
import torch.nn.functional as F
import torch.nn as nn
from network_ResNet import SEResNet
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np

def conv3x3(in_planes: int, out_planes: int, stride: int = 2, groups: int = 1, dilation: int = 1) -> torch.nn.Conv2d:
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        # SE
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(
            planes * 4, planes // 4, kernel_size=1, bias=False)
        self.conv_up = nn.Conv2d(
            planes // 4, planes * 4, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()
        # Downsample
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out1 = self.global_pool(out)
        out1 = self.conv_down(out1)
        out1 = self.relu(out1)
        out1 = self.conv_up(out1)
        out1 = self.sig(out1)

        if self.downsample is not None:
            residual = self.downsample(x)

        res = out1 * out + residual
        res = self.relu(res)

        return res

class MLP(torch.nn.Module):
    def __init__(self, n_output=1):
        super(MLP, self).__init__()
        ###############Resnet50
        self.view1 = SEResNet(Bottleneck, [3, 4, 6, 3])
        self.view2 = SEResNet(Bottleneck, [3, 4, 6, 3])
        ###############Resnet50


        self.model1 = maskrcnn_resnet50_fpn(pretrained = True)
        # 更换分类器
        self.in_features1 = self.model1.roi_heads.box_predictor.cls_score.in_features
        self.model1.roi_heads.box_predictor = FastRCNNPredictor(self.in_features1, num_classes = 2)
        

        self.model2 = maskrcnn_resnet50_fpn(pretrained = True)
        # 更换分类器
        self.in_features2 = self.model2.roi_heads.box_predictor.cls_score.in_features
        self.model2.roi_heads.box_predictor = FastRCNNPredictor(self.in_features2, num_classes = 2)
        



        self.out11 = torch.nn.Linear(65536, 3)  # output layer  4096
        self.out12 = torch.nn.Linear(65536, 3)  # output layer
        self.out13 = torch.nn.Linear(65536, 3)  # output layer



        self.out21 = torch.nn.Linear(65536, 3)  # output layer  4096
        self.out22 = torch.nn.Linear(65536, 3)  # output layer
        self.out23 = torch.nn.Linear(65536, 3)  # output layer

        self.out31 = torch.nn.Linear(65536, 3)  # output layer  4096
        self.out32 = torch.nn.Linear(65536, 3)  # output layer
        self.out33 = torch.nn.Linear(65536, 3)  # output layer
        # self.out1 = torch.nn.Sequential(
        #     torch.nn.Linear(4096,512),
        #     torch.nn.Dropout(p=0.5),
        #     torch.nn.Linear(512,3)
        # )
        # self.out2 = torch.nn.Sequential(
        #     torch.nn.Linear(4096,512),
        #     torch.nn.Dropout(p=0.5),
        #     torch.nn.Linear(512,3)
        # )
        # self.out3 = torch.nn.Sequential(
        #     torch.nn.Linear(4096,512),
        #     torch.nn.Dropout(p=0.5),
        #     torch.nn.Linear(512,3)
        # )



        # self.sigma1 = torch.nn.Parameter(torch.zeros(1))
        # self.sigma2 = torch.nn.Parameter(torch.zeros(1))
        # self.sigma3 = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x1,x2,x1_list,x2_list,target1,target2,typemode="train"):

        # maskrcnn
        ############
        lossoutput1 = self.model1(x1_list,target1)
        lossoutput2 = self.model2(x2_list,target2)
        if typemode == "train":
            self.model1.eval()
            self.model2.eval()
            with torch.no_grad():
                maskoutput1 = self.model1(x1_list,target1)
                maskoutput2 = self.model2(x2_list,target2)
            
            self.model1.train()
            self.model2.train()
        else:
            maskoutput1 = self.model1(x1_list,target1)
            maskoutput2 = self.model2(x2_list,target2)
        maskatt1 = torch.stack([torch.sum(x["masks"],dim=0) for x in maskoutput1[1]])
        maskatt2 = torch.stack([torch.sum(x["masks"],dim=0) for x in maskoutput2[1]])

        newx1 = maskatt1*x1
        newx2 = maskatt2*x2



        newx1 = self.view1(newx1)
        newx2 = self.view2(newx2)
        x = torch.cat((newx1, newx2), dim=1)

 

        ######手动归一化
        xout11 = self.out11(x)
        xout11new = torch.div(xout11,torch.linalg.norm(xout11,dim=1).unsqueeze(1))
        xout12 = self.out12(x)
        xout12new = torch.div(xout12,torch.linalg.norm(xout12,dim=1).unsqueeze(1))
        xout13 = self.out13(x)
        xout13new = torch.div(xout13,torch.linalg.norm(xout13,dim=1).unsqueeze(1))


        xout21 = self.out21(x)
        xout21new = torch.div(xout21,torch.linalg.norm(xout21,dim=1).unsqueeze(1))
        xout22 = self.out22(x)
        xout22new = torch.div(xout22,torch.linalg.norm(xout22,dim=1).unsqueeze(1))
        xout23 = self.out23(x)
        xout23new = torch.div(xout23,torch.linalg.norm(xout23,dim=1).unsqueeze(1))


        xout31 = self.out31(x)
        xout31new = torch.div(xout31,torch.linalg.norm(xout31,dim=1).unsqueeze(1))
        xout32 = self.out32(x)
        xout32new = torch.div(xout32,torch.linalg.norm(xout32,dim=1).unsqueeze(1))
        xout33 = self.out33(x)
        xout33new = torch.div(xout33,torch.linalg.norm(xout33,dim=1).unsqueeze(1))
        ######手动归一化


   
        # return xout1new,xout2new,xout3new
        return lossoutput1[0],lossoutput2[0],xout11new,xout12new,xout13new,xout21new,xout22new,xout23new,xout31new,xout32new,xout33new