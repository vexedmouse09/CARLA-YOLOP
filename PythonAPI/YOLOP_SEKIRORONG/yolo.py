#!/usr/bin/env python

import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont, Image

from nets.yolo import YoloBody, yoloR
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image)
from utils.utils_bbox import DecodeBox

from config import depth_img, project_seg

import cv2

Road = (128, 64, 128)
_RoadLine = (50, 234, 234)

'''
必读注释用于训练自己的数据集！
Mandatory comments for training your own dataset!
'''
class YOLO(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   When using your own trained model for prediction, you must modify model_path and classes_path!
        #   model_path points to the weights file in the logs folder, classes_path points to the txt under model_data
        #
        #   After training, there are multiple weight files in the logs folder, choose the one with lower validation loss.
        #   Lower validation loss does not mean higher mAP, only that this weight generalizes better on the validation set.
        #   If there is a shape mismatch, pay attention to modifying the model_path and classes_path parameters used during training.
        #--------------------------------------------------------------------------#
        "model_path"        : '/home/reu/carla/PythonAPI/YOLOP/Model/ep063-loss3.571-val_loss4.871.pth',
        "classes_path"      : '/home/reu/carla/PythonAPI/YOLOP/model_data/my_classes.txt',
        #---------------------------------------------------------------------#
        #   anchors_path represents the txt file corresponding to the prior box, usually does not need to be modified.
        #   anchors_mask is used to help the code find the corresponding prior box, usually does not need to be modified.
        #---------------------------------------------------------------------#
        "anchors_path"      : '/home/reu/carla/PythonAPI/YOLOP/model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #---------------------------------------------------------------------#
        #   The size of the input image must be a multiple of 32.
        #---------------------------------------------------------------------#
        "input_shape"       : [480, 640],
        #---------------------------------------------------------------------#
        #   Only prediction boxes with scores greater than the confidence threshold will be retained
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   nms_iou size used for non-maximum suppression
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.5,
        #---------------------------------------------------------------------#
        #   This variable is used to control whether to use letterbox_image for undistorted resize of the input image.
        #   After multiple tests, it is found that closing letterbox_image and directly resizing works better
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        #   Whether to use Cuda
        #   If you don't have a GPU, set it to False
        #-------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   Initialize YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        #   Get the number of categories and prior boxes
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (640, 480), self.anchors_mask)

        #---------------------------------------------------#
        #   Set different colors for drawing boxes
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    #---------------------------------------------------#
    #   Generate model
    #---------------------------------------------------#
    def generate(self):
        #---------------------------------------------------#
        #   Create YOLO model and load YOLO model weights
        #---------------------------------------------------#
        self.net    = yoloR(self.anchors_mask, self.num_classes)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    #---------------------------------------------------#
    #   Detect image
    #---------------------------------------------------#
    def detect_image(self, image, Dimage = None):
        #---------------------------------------------------#
        #   Calculate the height and width of the input image
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   Convert the image to RGB here to prevent grayscale images from causing errors during prediction.
        #   The code only supports RGB image prediction, all other types of images will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        if depth_img:
            Dimage = cvtColor(Dimage)
            Dimage_data = resize_image(Dimage, (640, 480), self.letterbox_image)
            Dimage_data = np.expand_dims(np.transpose(preprocess_input(np.array(Dimage_data, dtype='float32')), (2, 0, 1)), 0)
        #---------------------------------------------------------#
        #   Add gray bars to the image to achieve undistorted resize
        #   You can also directly resize for recognition
        #---------------------------------------------------------#
        image_data  = resize_image(image, (640,480), self.letterbox_image)
        #---------------------------------------------------------#
        #   Add batch_size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        road_mask = Image.new('RGB', (1280, 720), Road)
        road_mask = np.array(road_mask)

        ll_mask = Image.new('RGB', (1280, 720), _RoadLine)
        ll_mask = np.array(ll_mask)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            if depth_img:
                Dimages = torch.from_numpy(Dimage_data)
                if self.cuda:
                    Dimages = Dimages.cuda()

            #---------------------------------------------------------#
            #   Input the image into the network for prediction!
            #---------------------------------------------------------#
            if depth_img:
                outputs = self.net(images, Dimages)
            else:
                outputs = self.net(images)
            # print(outputs[3].shape)
            ll_predict = outputs[3].cpu()
            da_predict = outputs[4].cpu()
            outputs = outputs[0:3]
            outputs = self.bbox_util.decode_box(outputs)

            ll_predict *= 255
            da_predict *= 255

            ll_predict = np.array(ll_predict)
            da_predict = np.array(da_predict)

            ll_min = np.min(ll_predict)
            da_min = np.min(da_predict)
            ll_predict -= ll_min
            da_predict -= da_min
            ll_predict[ll_predict > 255] = 255
            da_predict[da_predict > 255] = 255
            ll = np.array(ll_predict, dtype=np.uint8)
            da = np.array(da_predict, dtype=np.uint8)

            ll = np.squeeze(ll)
            da = np.squeeze(da)
            ll = ll.transpose(1, 2, 0)
            da = da.transpose(1, 2, 0)
            ll = cv2.resize(ll, (1280, 720), interpolation=cv2.INTER_NEAREST)
            da = cv2.resize(da, (1280, 720), interpolation=cv2.INTER_NEAREST)

            da = cv2.cvtColor(da, cv2.COLOR_BGR2GRAY)
            retval, da = cv2.threshold(da, 0, 255, cv2.THRESH_OTSU)
            da_rgb = cv2.cvtColor(da, cv2.COLOR_GRAY2RGB)

            ll = cv2.cvtColor(ll, cv2.COLOR_BGR2GRAY)
            retval, ll = cv2.threshold(ll, 0, 255, cv2.THRESH_OTSU)
            ll_rgb = cv2.cvtColor(ll, cv2.COLOR_GRAY2RGB)

            da_rgb = cv2.bitwise_and(da_rgb, road_mask)
            ll_rgb = cv2.bitwise_and(ll_rgb, ll_mask)

            #---------------------------------------------------------#
            #   Stack the prediction boxes, then perform non-maximum suppression
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                        image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)

            if results[0] is None:
                image = np.array(image, dtype=np.uint8)

                if project_seg:
                    image = cv2.addWeighted(da_rgb, 0.42, image, 0.84, 0)
                    image = cv2.addWeighted(ll_rgb, 0.3, image, 0.84, 0)
                return image

            top_label   = np.array(results[0][:, 6], dtype='int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
        #---------------------------------------------------------#
        #   Set font and border thickness
        #---------------------------------------------------------#
        # font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        #---------------------------------------------------------#
        #   Image drawing
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(719, np.floor(bottom).astype('int32'))
            right   = min(1279, np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)

            label = label.encode('utf-8')

            image = np.array(image, dtype=np.uint8)

            image = cv2.rectangle(image, (left, bottom), (right, top), self.colors[c], 2)
            image = cv2.putText(image, str(label, 'UTF-8'), (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, self.colors[c], 2)

        if project_seg:
            image = cv2.addWeighted(da_rgb, 0.42, image, 0.84, 0)
            image = cv2.addWeighted(ll_rgb, 0.3, image, 0.84, 0)

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   Convert the image to RGB here to prevent grayscale images from causing errors during prediction.
        #   The code only supports RGB image prediction, all other types of images will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Add gray bars to the image to achieve undistorted resize
        #   You can also directly resize for recognition
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   Add batch_size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   Input the image into the network for prediction!
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   Stack the prediction boxes, then perform non-maximum suppression
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                                                    
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------------#
                #   Input the image into the network for prediction!
                #---------------------------------------------------------#
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                #---------------------------------------------------------#
                #   Stack the prediction boxes, then perform non-maximum suppression
                #---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                            
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w") 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   Convert the image to RGB here to prevent grayscale images from causing errors during prediction.
        #   The code only supports RGB image prediction, all other types of images will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Add gray bars to the image to achieve undistorted resize
        #   You can also directly resize for recognition
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   Add batch_size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   Input the image into the network for prediction!
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   Stack the prediction boxes, then perform non-maximum suppression
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 6], dtype='int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return 
