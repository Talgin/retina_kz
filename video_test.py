import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface import RetinaFace

from imutils.video import FileVideoStream
from imutils.video import FPS

count = 1

gpuid = 0
detector = RetinaFace('/home/ti/Downloads/MODELS/retinaface-R50/R50', 0, gpuid, 'net3')


cap = cv2.VideoCapture('rtsp://admin:Admin123!@10.150.30.202:554/live')
# start the FPS timer
fps = FPS().start()

while(cap.isOpened()):

  thresh = 0.8
  scales = [1080, 1920]# [1024, 1980]  

  ret, img = cap.read()
  # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  #print(frame.shape)
  #cv2.imshow('frame', img)

  #img = cv2.imread(gray)
  print(img.shape)
  print(scales[1])
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

  print('im_scale', im_scale)

  scales = [im_scale]
  flip = False

  for c in range(count):
    faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
    print(c, faces.shape, landmarks.shape)

  if faces is not None:
    #if faces.shape > 0:
    #  print('find', faces.shape[0], 'faces')
    print('find', faces.shape[0], 'faces')
    for i in range(faces.shape[0]):
      #print('score', faces[i][4])
      box = faces[i].astype(np.int)
      #color = (255,0,0)
      color = (0,0,255)
      cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
      if landmarks is not None:
        landmark5 = landmarks[i].astype(np.int)
        #print(landmark.shape)
        for l in range(landmark5.shape[0]):
          color = (0,0,255)
          if l==0 or l==3:
            color = (0,255,0)
          cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

    filename = './detector_test.jpg'
    print('writing', filename)
    cv2.imshow(filename, img)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  fps.update()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
cap.release()
cv2.destroyAllWindows()
