import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F



def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: 
    #For confidence (class labels), use cross entropy (F.cross_entropy)
    #You can try F.binary_cross_entropy and see which loss is better
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.

    # Reshape
    B, N, C = pred_confidence.shape
    pred_confidence = pred_confidence.view(B*N, C)
    ann_confidence = ann_confidence.view(B*N, C)
    ann_box = ann_box.view(B*N, C)
    pred_box = pred_box.view(B*N, C)
    
    # Filter
    non_empty_ann_confidence = ann_confidence[ann_confidence[:, 3] != 1]
    non_empty_pred_confidence = pred_confidence[ann_confidence[:, 3] != 1]
    non_empty_bbox = ann_box[ann_confidence[:, 3] != 1]
    non_empty_pred_bbox = pred_box[ann_confidence[:, 3] != 1]
    empty_ann_confidence = ann_confidence[ann_confidence[:, 3] == 1]
    empty_pred_confidence = pred_confidence[ann_confidence[:, 3] == 1]
    
    # Compute Loss
    class_loss = F.cross_entropy(non_empty_pred_confidence, non_empty_ann_confidence) + 3 * F.cross_entropy(empty_ann_confidence, empty_pred_confidence)
    bbox_loss = F.smooth_l1_loss(non_empty_pred_bbox, non_empty_bbox)
    loss = class_loss + bbox_loss

    return loss


class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()

        self.class_num = class_num

        self.conv_layers_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(num_features=64), # TODO: confirm num feat??
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )  

        self.conv_layers_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )

        self.conv_layers_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )

        self.conv_layers_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )

        self.conv_res5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )

        self.conv_res3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )

        self.conv_res1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )

        self.conv_res10_o1 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_res10_o2 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_res5_o1 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_res5_o2 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_res3_o1 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_res3_o2 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_res1_o1 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_res1_o2 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x):

        x = x/255.0 #normalize image.

        x = self.conv_layers_1(x)
        x = self.conv_layers_2(x)
        x = self.conv_layers_3(x)

        res10 = self.conv_layers_4(x)
        res5 = self.conv_res5(res10)
        res3 = self.conv_res3(res5)
        res1 = self.conv_res1(res3)

        B, _, H, W = res10.shape
        res10_o1 = self.conv_res10_o1(res10).view(B, 16, H*W)
        res10_o2 = self.conv_res10_o2(res10).view(B, 16, H*W)

        B, _, H, W = res5.shape
        res5_o1 = self.conv_res5_o1(res5).view(B, 16, H*W)
        res5_o2 = self.conv_res5_o2(res5).view(B, 16, H*W)

        B, _, H, W = res3.shape
        res3_o1 = self.conv_res3_o1(res3)
        # print(res3_o1[0,1,:,:])
        res3_o1 = res3_o1.view(B, 16, H*W)
        # print(res3_o1[0,1,:])
        res3_o2 = self.conv_res3_o2(res3).view(B, 16, H*W)

        B, _, H, W = res1.shape
        res1_o1 = self.conv_res1_o1(res1).view(B, 16, H*W)
        res1_o2 = self.conv_res1_o2(res1).view(B, 16, H*W)

        out_bbox = torch.concat([res10_o1, res5_o1, res3_o1, res1_o1], axis=2)
        B, H, W = out_bbox.shape
        out_bbox = out_bbox.view(B, W, H)
        bboxes = out_bbox.view(B, 540, 4)

        out_conf = torch.concat([res10_o2, res5_o2, res3_o2, res1_o2], axis=2)
        B, H, W = out_conf.shape
        out_conf = out_conf.view(B, W, H)
        out_conf = out_conf.view(B, 540, 4)
        confidence = torch.softmax(out_conf, dim=2)
        
        print(confidence.shape, bboxes.shape)
            
        return confidence,bboxes










