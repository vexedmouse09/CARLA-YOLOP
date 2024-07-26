#!/usr/bin/env python
# Attempted to create this to create an anchor file....

import numpy as np
from sklearn.cluster import KMeans

def iou(box, clusters):
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_

def avg_iou(boxes, clusters):
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

def kmeans(boxes, k, dist=np.median):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(boxes)
    return kmeans.cluster_centers_

def load_dataset(path):
    dataset = []
    with open(path) as f:
        for line in f:
            annotations = line.strip().split()[1:]
            for annotation in annotations:
                x_min, y_min, x_max, y_max, _ = map(float, annotation.split(','))
                width = x_max - x_min
                height = y_max - y_min
                dataset.append([width, height])
    return np.array(dataset)

def generate_anchors(annotation_file, num_clusters):
    dataset = load_dataset(annotation_file)
    anchors = kmeans(dataset, num_clusters)
    print("Anchors: ", anchors)
    print("Average IOU: ", avg_iou(dataset, anchors))
    return anchors

if __name__ == '__main__':
    annotation_file = '/home/reu/carla/PythonAPI/YOLOP/annotations.txt'
    num_clusters = 9
    anchors = generate_anchors(annotation_file, num_clusters)
    with open('/home/reu/carla/PythonAPI/YOLOP/anchors.txt', 'w') as f:
        anchor_strings = ', '.join([f"{int(w)},{int(h)}" for w, h in anchors])
        f.write(anchor_strings)
        f.write('\n')