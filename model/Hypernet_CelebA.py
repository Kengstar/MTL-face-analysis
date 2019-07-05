import torch
import torchvision.models as models
import torch.nn as nn
from model.task_branch import *


class HyperNet_CelebA(nn.Module):
    def __init__(self, tasks, attributes, init_variances, nr_of_landmarks):
        super(HyperNet_CelebA, self).__init__()
        resnet = models.resnet101(pretrained=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task_branches = nn.ModuleList()
        self.uncertainties = []
        self.tasks = tasks + attributes
        self.attributes = attributes

        for init_var in init_variances:
            self.uncertainties.append(torch.tensor(init_var, requires_grad=True, device=device))
        ####Encoder

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = resnet.maxpool
        self.sigmoid = nn.Sigmoid()
        self.layer1 = resnet.layer1

        #feature fusion
        self.side_conv1 = nn.Sequential(nn.Conv2d(256, 256, stride=2, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 512, kernel_size=1), nn.BatchNorm2d(512), nn.ReLU())
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.side_conv2 = nn.Sequential(nn.Conv2d(512, 512, stride=2, kernel_size=3, padding=1), nn.BatchNorm2d(512),
                                        nn.ReLU(),
                                        nn.Conv2d(512, 1024, kernel_size=1),nn.BatchNorm2d(1024), nn.ReLU())
        self.side_conv3 = nn.Sequential(nn.Conv2d(1024, 1024, stride=2, kernel_size=3, padding=1), nn.BatchNorm2d(1024),
                                        nn.ReLU(),
                                        nn.Conv2d(1024, 2048, kernel_size=1), nn.BatchNorm2d(2048), nn.ReLU())
        self.layer4 = resnet.layer4

        # global average pooling, could also be done with adaptive average pooling
        self.avgpool = nn.AvgPool2d(7, stride=1)
        nr_of_features = 2048

        # task branches
        self.task_branches.append(TaskBranchLinearOutput(in_nodes=nr_of_features, hidden_nodes=512,
                                                            out_nodes=nr_of_landmarks * 2))
        self.task_branches.append(TaskBranchLinearOutput(in_nodes=nr_of_features, hidden_nodes=512,
                                                            out_nodes=4))

        self.task_branches.append(TaskBranchAttributes(in_nodes=nr_of_features, hidden_nodes=512,
                                                                  out_nodes=len(attributes)))

        ##TODO nn.sequential
        self.up1 = nn.ConvTranspose2d(in_channels=2048, out_channels=64, kernel_size=3, stride=2,
                                      padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2,
                                      padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2,
                                      padding=1, output_padding=1)
        self.up4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2,
                                      padding=1, output_padding=1)
        self.up5 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=2,
                                      padding=1, output_padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for l_idx, layer in enumerate(self.layer1):
            x = layer(x)
            if l_idx == len(self.layer1)-1:
                res2c = x

        hyper_features = self.side_conv1(res2c)

        for l_idx, layer in enumerate(self.layer2):
            x = layer(x)
            if l_idx == len(self.layer2)-1:
                res3b3 = x

        hyper_features = hyper_features + res3b3
        hyper_features = self.side_conv2(hyper_features)

        for l_idx, layer in enumerate(self.layer3):
            x = layer(x)
            if l_idx == len(self.layer3)-1:
                res4b22 = x

        hyper_features = hyper_features + res4b22
        hyper_features = self.side_conv3(hyper_features)

        for l_idx, layer in enumerate(self.layer4):
            x = layer(x)
            if l_idx == len(self.layer4)-1:
                res5c = x

        hyper_features = hyper_features + res5c
        x = self.avgpool(hyper_features)
        x = x.view(x.size(0), -1)

        pred_lm = self.task_branches[0](x)
        pred_bbox = self.task_branches[1](x)        ##module list for loop
        pred_attr = self.task_branches[2](x)
                                                    #can be reduced into for loop
        pred_seg = self.up1(hyper_features)         ### nn.sequential
        pred_seg = self.up2(pred_seg)
        pred_seg = self.up3(pred_seg)
        pred_seg = self.up4(pred_seg)
        pred_seg = self.up5(pred_seg)

        return pred_lm, pred_bbox, pred_attr, pred_seg
