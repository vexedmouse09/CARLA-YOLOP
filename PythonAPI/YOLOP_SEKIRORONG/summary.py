#!/usr/bin/env python

#--------------------------------------------#
#   This part of the code is used to view the network structure
#--------------------------------------------#
import torch
from torchsummary import summary

from nets.yolo import YoloBody

if __name__ == "__main__":
    # You need to use device to specify whether the network runs on GPU or CPU
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m       = YoloBody([[6, 7, 8], [3, 4, 5], [0, 1, 2]], 80).to(device)
    summary(m, input_size=(3, 416, 416))