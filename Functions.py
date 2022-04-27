# This file contains the CONSTANTS and functions that are used in other notebooks

import cv2
import numpy as np
import glob, os
from os import listdir
import math
from matplotlib import pyplot as plt
import time
import random


# get label from a given path
# filename: 'xxxxxxxx.txt'
# path: path to the folder containing the file
def get_label(filename, path):
    labels = []
    f = open(path+filename, "r")
    for line in f.readlines():
        line = line[:-1]
        arr = filter(None, line.split(' '))
        arr = list(map(float, arr))
        arr[0] = int(arr[0])
        labels.append(arr)
    return labels


# convert label's x and y from center to topleft of bounding box
# for image visualization
def center_to_topleft(label):
    return [label[0]-label[2]/2, label[1]-label[3]/2, label[2], label[3]]


# unnormalize label
# for processing label before image visualization
def unnormalize(labels, width, height):
    return_labels = []
    for label in labels:
        return_labels.append(center_to_topleft([label[1]*width, label[2]*height, label[3]*width, label[4]*height]))
    return return_labels


# get width and height of an image
def get_wh(path):
    img = cv2.imread(path)
    return img.shape[1], img.shape[0]


# size of label
def size(label, width, height):
    return int(label[3]*width*label[4]*height)


#Normalize and change to center coordinate
def normalize(label, width, height):   
    return_label = [8, (label[0]+label[2]/2)/width, (label[1]+label[3]/2)/height, label[2]/width, label[3]/height]
    return_label = [ round(elem, 6) for elem in return_label ]
    return return_label


# calculation of overlapping ratio in IoU
def overlapping(label, pred):
    x1 = label[1]
    y1 = label[2]
    w1 = label[3]
    h1 = label[4]
    
    x2 = pred[1]
    y2 = pred[2]
    w2 = pred[3]
    h2 = pred[4]
    
    XA1 = x1
    XA2 = x1 + w1
    YA1 = y1
    YA2 = y1 + h1
    XB1 = x2
    XB2 = x2 + w2
    YB1 = y2
    YB2 = y2 + h2
    
    SI = max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1))
    SA = w1 * h1
    SB = w2 * h2
    SU = SA + SB - SI

    return SI / SU

# filter prediction with confidence higher than the confidence threshold
def filter_preds(preds, CONF_THRESHOLD):
    return list(filter(lambda x: x[5] >= CONF_THRESHOLD, preds))


# reference: https://colab.research.google.com/github/albumentations-team/albumentations_examples/blob/colab/example_bboxes2.ipynb#scrollTo=UCiQvSMD-Ks9

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

# image visualization with bounding box
def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


# image visualization with bounding box
def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    
    
    