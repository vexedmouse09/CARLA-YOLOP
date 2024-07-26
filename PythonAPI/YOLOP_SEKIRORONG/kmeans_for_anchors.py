#!/usr/bin/env python
# GENERATES YOLO ANCHORS
#-------------------------------------------------------------------------------------------------#
#   Although kmeans will cluster the boxes in the dataset, many datasets have similar box sizes,
#   resulting in clustered 9 boxes that are not much different. Such boxes are not conducive to
#   model training. Different feature layers are suitable for different sizes of prior boxes.
#   Shallower feature layers are suitable for larger prior boxes. The prior boxes of the original
#   network have been allocated in large, medium, and small proportions, and the effect will be
#   very good without clustering.
#-------------------------------------------------------------------------------------------------#
import glob
import xml.etree.ElementTree as ET

import numpy as np

def bbox3Dtowh(bboxs, data):
    bbox2d = []
    for bbox in bboxs:
        bbox_class = bbox[0]
        bbox3d = bbox[1]
        x, y, x2, y2 = bbox3d[8][0], bbox3d[8][1], bbox3d[9][0], bbox3d[9][1]
        if x == y == x2 == y2 == 0:
            continue
        data.append([x2 - x, y2 - y])

def parse3Dbbox(path, data):
    with open(path, 'r') as f:
        labels = []
        label = []
        point = []
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split(' ')
            label.append(line[0])
            for i in range(int(len(line) / 2) - 1):
                point.append((int(line[2 * i + 1]), int(line[2 * i + 2])))
            label.append(point)
            point = []
            labels.append(label)
            label = []

    bbox3Dtowh(labels, data)

def cas_iou(box, cluster):
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:, 0] * cluster[:, 1]
    iou = intersection / (area1 + area2 - intersection)

    return iou

def avg_iou(box, cluster):
    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])

def kmeans(box, k):
    #-------------------------------------------------------------#
    #   Get the total number of boxes
    #-------------------------------------------------------------#
    row = box.shape[0]
    
    #-------------------------------------------------------------#
    #   Positions of each point in each box
    #-------------------------------------------------------------#
    distance = np.empty((row, k))
    
    #-------------------------------------------------------------#
    #   Final clustering positions
    #-------------------------------------------------------------#
    last_clu = np.zeros((row,))

    np.random.seed()

    #-------------------------------------------------------------#
    #   Randomly select 5 as clustering centers
    #-------------------------------------------------------------#
    cluster = box[np.random.choice(row, k, replace=False)]
    while True:
        #-------------------------------------------------------------#
        #   Calculate the iou situation of each row with five points.
        #-------------------------------------------------------------#
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i], cluster)
        
        #-------------------------------------------------------------#
        #   Get the minimum point
        #-------------------------------------------------------------#
        near = np.argmin(distance, axis=1)

        if (last_clu == near).all():
            break
        
        #-------------------------------------------------------------#
        #   Find the median point of each class
        #-------------------------------------------------------------#
        for j in range(k):
            cluster[j] = np.median(
                box[near == j], axis=0)

        last_clu = near

    return cluster

def load_data(path):
    data = []

    for filename in (glob.glob(path + '/*.txt')):
        parse3Dbbox(filename, data)

    return np.array(data)

if __name__ == '__main__':
    #-------------------------------------------------------------#
    #   Running this program will calculate the xml in './VOCdevkit/VOC2007/Annotations'
    #   and generate yolo_anchors.txt
    #-------------------------------------------------------------#
    SIZE_x = 640
    SIZE_y = 480
    anchors_num = 9
    #-------------------------------------------------------------#
    #   Load the dataset, you can use VOC's xml
    #-------------------------------------------------------------#
    path = r'G:\Carla_Dataset\3Dbbox\train'
    
    #-------------------------------------------------------------#
    #   Load all xmls
    #   Stored format is width, height converted to ratio
    #-------------------------------------------------------------#
    data = load_data(path)
    
    #-------------------------------------------------------------#
    #   Use k-means clustering algorithm
    #-------------------------------------------------------------#
    out = kmeans(data, anchors_num)
    out = out[np.argsort(out[:, 0])]
    print('acc:{:.2f}%'.format(avg_iou(data, out) * 100))
    print(out * SIZE_x)
    data = out
    f = open("yolo_anchors.txt", 'w')
    row = np.shape(data)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (data[i][0], data[i][1])
        else:
            x_y = ", %d,%d" % (data[i][0], data[i][1])
        f.write(x_y)
    f.close()

