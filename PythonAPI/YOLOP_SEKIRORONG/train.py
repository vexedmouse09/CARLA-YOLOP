#!/usr/bin/env python

#-------------------------------------#
#       Train the dataset
#-------------------------------------#
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody, yoloR
from nets.yolo_training import YOLOLoss, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes
from utils.utils_fit import fit_one_epoch

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

'''
When training your own object detection model, you must pay attention to the following points:
1. Carefully check whether your format meets the requirements before training. This library requires the dataset format to be in VOC format. The prepared content should include input images and labels.
   The input images are .jpg images, and no fixed size is required. They will be automatically resized before being passed into training.
   Grayscale images will be automatically converted to RGB images for training, no need to modify them yourself.
   If the suffix of the input image is not jpg, you need to batch convert them to jpg before starting training.

   The labels are in .xml format, and the label files correspond to the input image files.

2. The trained weight files are saved in the logs folder, and one weight file is saved for each epoch. If you only trained for a few steps, the weights will not be saved. You need to clarify the concepts of epoch and step.
   During training, the code does not set to only save the lowest loss. Therefore, there will be 100 weights if you train with the default parameters. If there is not enough space, you can delete some yourself.
   It's not better to save more or less. Some people want to save all, and some want to save only a few. To meet most needs, it's better to save all for higher selectivity.

3. The size of the loss value is used to determine whether it converges. The more important thing is the trend of convergence, that is, the validation set loss keeps decreasing. If the validation set loss basically does not change, the model is basically converged.
   The specific size of the loss value does not mean anything. Large or small depends on the calculation method of the loss, not that close to 0 is good. If you want to make the loss look better, you can divide it by 10000 in the corresponding loss function.
   The loss values during the training process will be saved in the logs folder under the loss_%Y_%m_%d_%H_%M_%S folder.

4. Parameter tuning is quite important. There are no absolutely good parameters. The existing parameters are tested to be able to train normally. Therefore, I would suggest using the existing parameters.
   But the parameters themselves are not absolute. For example, with the increase of batch size, the learning rate can also be increased, and the effect will be better; do not use too large a learning rate for too deep a network, etc.
   These are based on experience and can only be gained by consulting more materials and trying it yourself.
'''  
if __name__ == "__main__":
    #-------------------------------#
    #   Whether to use Cuda
    #   If you don't have a GPU, set it to False
    #-------------------------------#
    Cuda = True
    #--------------------------------------------------------#
    #   Be sure to modify classes_path before training to correspond to your own dataset
    #--------------------------------------------------------#
    classes_path    = '/home/reu/carla/PythonAPI/YOLOP/model_data/my_classes.txt'
    #---------------------------------------------------------------------#
    #   anchors_path represents the txt file corresponding to the prior box, usually does not need to be modified.
    #   anchors_mask is used to help the code find the corresponding prior box, usually does not need to be modified.
    #---------------------------------------------------------------------#
    anchors_path    = '/home/reu/carla/PythonAPI?YOLOP/model_data/yolo_anchors.txt'
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    #----------------------------------------------------------------------------------------------------------------------------#
    #   For the download of the weight file, please see the README, which can be downloaded through the cloud disk. The pre-trained weights of the model are universal for different datasets because the features are universal.
    #   The important part of the pre-trained weights of the model is the weight part of the backbone feature extraction network, which is used for feature extraction.
    #   Pre-trained weights must be used in 99% of cases. If not used, the weights of the backbone part are too random, the feature extraction effect is not obvious, and the network training result will not be good.
    #
    #   If there is an interruption in training, you can set the model_path to the weight file in the logs folder and reload the weights that have already been partially trained.
    #   At the same time, modify the parameters of the frozen stage or the unfrozen stage below to ensure the continuity of the model epoch.
    #
    #   When model_path = '', the weights of the entire model are not loaded.
    #
    #   Here, the weights of the entire model are used, so they are loaded in train.py.
    #   If you want to train the model from scratch, set model_path = '', and set Freeze_Train = False below. In this case, the training starts from scratch, and there is no process of freezing the backbone.
    #   Generally speaking, training from scratch will have poor results because the weights are too random, and the feature extraction effect is not obvious.
    #
    #   Networks are generally not trained from scratch, at least the weights of the backbone part will be used. Some papers mention that pre-training is not necessary, mainly because their dataset is large and they have excellent parameter tuning ability.
    #   If you must train the backbone part of the network, you can understand the imagenet dataset, first train the classification model, the backbone part of the classification model is common to this model, and train based on this.
    #----------------------------------------------------------------------------------------------------------------------------#
    # model_path      = 'model_data/yolo4_weights.pth'
    model_path      = '/home/reu/carla/PythonAPI/YOLOP/Model/ep063-loss3.571-val_loss4.871.pth'
    #------------------------------------------------------#
    # model_path = ''
    #   The input shape size must be a multiple of 32
    #------------------------------------------------------#
    input_shape     = [480, 640]
    #------------------------------------------------------#
    #   Yolov4's tricks application
    #   mosaic data enhancement True or False 
    #   In actual tests, mosaic data enhancement is not stable, so it is set to False by default
    #   Cosine_lr cosine annealing learning rate True or False
    #   label_smoothing label smoothing generally below 0.01, such as 0.01, 0.005
    #------------------------------------------------------#
    mosaic              = False
    Cosine_lr           = False
    label_smoothing     = 0

    #----------------------------------------------------#
    #   Training is divided into two phases: frozen phase and unfrozen phase.
    #   Insufficient memory has nothing to do with the dataset size. If you are prompted that the memory is insufficient, please reduce the batch_size.
    #   Affected by the BatchNorm layer, the batch_size must be at least 2, and cannot be 1.
    #----------------------------------------------------#
    #----------------------------------------------------#
    #   Parameters for frozen phase training
    #   At this time, the backbone of the model is frozen, and the feature extraction network does not change
    #   It occupies less memory and only fine-tunes the network
    #----------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 100
    Freeze_batch_size   = 8
    Freeze_lr           = 1e-5
    #----------------------------------------------------#
    #   Parameters for unfrozen phase training
    #   At this time, the backbone of the model is not frozen, and the feature extraction network will change
    #   It occupies more memory, and all parameters of the network will change
    #----------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 4
    Unfreeze_lr         = 1e-4
    #------------------------------------------------------#
    #   Whether to perform frozen training, by default, the backbone is trained first and then unfrozen for training.
    #------------------------------------------------------#
    Freeze_Train        = True
    #------------------------------------------------------#
    #   Used to set whether to use multi-threaded data loading
    #   Turning it on will speed up data loading, but it will take up more memory
    #   Computers with less memory can be set to 2 or 0  
    #------------------------------------------------------#
    num_workers         = 8
    #----------------------------------------------------#
    #   Get the image path and labels
    #----------------------------------------------------#
    dataset_img_path = r"/home/reu/Carla_Dataset/Image"

    train_annotation_path   = dataset_img_path + '/train/images.txt'
    val_annotation_path     = dataset_img_path + '/val/images.txt'

    #----------------------------------------------------#
    #   Get classes and anchors
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    #------------------------------------------------------#
    #   Create yolo model
    #------------------------------------------------------#
    model = yoloR(anchors_mask, num_classes)
    weights_init(model, init_type='kaiming')
    if model_path != '':
        #------------------------------------------------------#
        #   For the weight file, please see the README, Baidu Netdisk download
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    yolo_loss    = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing)
    loss_history = LossHistory("logs/")

    #---------------------------#
    #   Read the txt corresponding to the dataset
    #---------------------------#
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    #------------------------------------------------------#
    #   The features of the backbone feature extraction network are universal. Freezing training can speed up training
    #   It can also prevent the weights from being destroyed in the early stage of training.
    #   Init_Epoch is the initial epoch
    #   Freeze_Epoch is the epoch for frozen training
    #   UnFreeze_Epoch is the total training epoch
    #   If prompted with OOM or insufficient memory, please reduce Batch_size
    #------------------------------------------------------#
    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch
                        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to train, please expand the dataset.")
        
        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset   = YoloDataset(train_annotation_path, input_shape, num_classes, mosaic=mosaic, train = True)
        val_dataset     = YoloDataset(val_annotation_path, input_shape, num_classes, mosaic=False, train = False)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)

        #------------------------------------#
        #   Train with part of the model frozen
        #------------------------------------#
        if Freeze_Train:
            for param in model.backbone_DetectHead.parameters():
                param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()
            
    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch
                        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to train, please expand the dataset.")
        
        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr.scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset   = YoloDataset(train_annotation_path, input_shape, num_classes, mosaic=mosaic, train = True)
        val_dataset     = YoloDataset(val_annotation_path, input_shape, num_classes, mosaic=False, train = False)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)

        #------------------------------------#
        #   Train with part of the model unfrozen
        #------------------------------------#
        if Freeze_Train:
            for param in model.backbone_DetectHead.parameters():
                param.requires_grad = True

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()

