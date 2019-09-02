# Testing one head crop
import cv2
import sys
import numpy as np
import datetime
import os
import glob
from imutils.video import FileVideoStream
from imutils.video import FPS
from imutils.video import WebcamVideoStream
import imutils
import time
from retinaface import RetinaFace
import skimage

thresh = 0.8
scales = [1024, 1980]

count = 1

gpuid = 0
detector = RetinaFace('/home/ti/Downloads/MODELS/retinaface-R50/R50', 0, gpuid, 'net3')

img = cv2.imread('./detections/2019_08_08 17_01_03_224213.jpg')
# print(img.shape)
im_shape = img.shape
target_size = scales[0]
max_size = scales[1]
im_size_min = np.min(im_shape[0:2])
im_size_max = np.max(im_shape[0:2])
#im_scale = 1.0
#if im_size_min>target_size or im_size_max>max_size:
im_scale = float(target_size) / float(im_size_min)
# prevent bigger axis from being more than max_size:
if np.round(im_scale * im_size_max) > max_size:
    im_scale = float(max_size) / float(im_size_max)

# print('im_scale', im_scale)

scales = [im_scale]
flip = False

for c in range(count):
  faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
  # print(c, faces.shape, landmarks.shape)

if faces is not None:  
  print('find', faces.shape[0], 'faces')
  for i in range(faces.shape[0]):
    #print('score', faces[i][4])
    box = faces[i].astype(np.int)
    print(box)
    #color = (255,0,0)
    color = (0,255,0)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)    
    if landmarks is not None:
      landmark5 = landmarks[i].astype(np.int)
      #print(landmark.shape)
      for l in range(landmark5.shape[0]):
        color = (0,0,255)
        if l==0 or l==3:
          color = (0,255,0)
        cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
    # Calculating cropping area
    img_size = 112
    center_y = box[1] + ((box[3] - box[1])/2) # calculating center of the x side
    center_x = box[0] + ((box[2] - box[0])/2) # calculating center of the y side
    rect_y = center_y - img_size/2  # calculating starting x of rectangle
    rect_x = center_x - img_size/2  # calculating starting y of rectangle     
    # Cropping an area
    ret_img = img[rect_y:rect_y+112,rect_x:rect_x+112]
    fname = './crop_test.jpg'    
    cv2.imwrite(fname, ret_img)

  filename = './detector_test.jpg'
  print('writing', filename)
  cv2.imwrite(filename, img)

