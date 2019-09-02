# This is faster version of video capture and face detection module that utilizes
# RetinaFace face detector
# To use just run python video_test_fast.py 
# You can change face detection model on the 24-th line changing the first parameter for detector
# Module uses Adrian Rosebrock's imutils library for faster video processing
import cv2
import sys
import numpy as np
import datetime
import os, time
import glob
from imutils.video import FileVideoStream
from imutils.video import FPS
from imutils.video import WebcamVideoStream
import imutils
import time
from retinaface import RetinaFace
import skimage
import threading

count = 1

gpuid = 0
# Loading model for face detection
detector = RetinaFace('/home/ti/Downloads/MODELS/retinaface-R50/R50', 0, gpuid, 'net3')
# Reading stream from camera
fvs = WebcamVideoStream(src='rtsp://admin:Admin123!@10.150.30.202:554/live').start()
time.sleep(1.0)

# Start fps counter
fps = FPS().start()

cnt = 0
# Main loop
while True:
  img = fvs.read()

  thresh = 0.8
  scales = [1080, 1920] # [1024, 1980]   
    
  print(img.shape)
  # print(scales[1])
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

  scales = [im_scale]
  flip = False

  for c in range(count):
    faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)

  if faces is not None:
    print('find', faces.shape[0], 'faces')
    for i in range(faces.shape[0]):
      #print('score', faces[i][4])
      box = faces[i].astype(np.int)
      #color = (255,0,0)
      color = (0,255,0)
      print(box)

      filename = str(datetime.datetime.now()).replace(":", "_").replace(".", "_").replace("-", "_").replace(' ', '_') + '.jpg'

      cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)  # simple rectangle around face

      font = cv2.FONT_HERSHEY_SIMPLEX
      text = 'x: ' + str(box[0]) + ';  y: ' + str(box[1]) # + ' ' + str(box[2]) + ' ' + str(box[3])
      cv2.putText(img,text,(50,50), font, 1, (0,255,255), 2, cv2.LINE_AA)
      if landmarks is not None:
        landmark5 = landmarks[i].astype(np.int)
        #print(landmark.shape)
        for l in range(landmark5.shape[0]):
          color = (0,0,255)
          if l==0 or l==3:
            color = (0,255,0)
          cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)    

      
      print('Faces found', filename)
      # cv2.imwrite('./cropped/' + filename, scaled)
      cv2.imwrite('/home/ti/Downloads/SERVER_CODE/shots/' + filename, img)
      # break   
    cv2.imshow('image', img)          
    
  # update fps counter
  fps.update()

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# cap.release()
cv2.destroyAllWindows()
fvs.stop()
