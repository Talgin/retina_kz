# This is faster version of video capture and face detection module that utilizes
# RetinaFace face detector
# To use just run python video_test_fast.py 
# You can change face detection model on the 22-nd line changing the first parameter for detector
# Module uses Adrian Rosebrock's imutils library for faster video processing
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
import face_model
import argparse
import comparator_class
import get_results

parser = argparse.ArgumentParser(description='Extracting input image features')
#general
parser.add_argument('--image-size', default='112,112', help='Image size in pixels')
# parser.add_argument('--model', default='/home/ti/Downloads/insightface/deploy/models/model-r100-ii/model,0', help='path to load model.') 
parser.add_argument('--client-features', default='', help='Features from client side app')
parser.add_argument('--server-features', default='', help='Features from server (database or some json file)')
parser.add_argument('--model', default='/home/ti/Downloads/SERVER_CODE/models/kaz/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--input-type', default=0, help='Type of input: 0 - json, 1 - array')
args = parser.parse_args()
input_type = args.input_type
comp = comparator_class.Comparator(args)

count = 1

gpuid = 0
# Loading model for face detection
detector = RetinaFace('/home/ti/Downloads/MODELS/retinaface-R50/R50', 0, gpuid, 'net3')
# Reading stream from camera
fvs = WebcamVideoStream(src='rtsp://admin:Admin123!@10.150.30.202:554/live').start()
time.sleep(1.0)

# Loading models
extractor = '/home/ti/Downloads/SERVER_CODE/models/model-r100-arcface-ms1m-refine-v2/model-r100-ii/model,0'
model = face_model.FaceModel(args)

# Start fps counter
fps = FPS().start()

cnt = 0

# Main loop
while True:
  img = fvs.read()

  thresh = 0.8
  scales = [1080, 1920] # [1024, 1980]   
    
  print(img.shape)

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
    # print(c, faces.shape, landmarks.shape)

  if faces is not None:
    print('find', faces.shape[0], 'faces')
    for i in range(faces.shape[0]):
      #print('score', faces[i][4])
      box = faces[i].astype(np.int)
      #color = (255,0,0)
      color = (0,255,0)  

      filename = str(datetime.datetime.now()).replace(":", "_").replace(".", "_").replace("-", "_") + '.jpg'
      
      # Calculate cropping area
      img_size = 112
      center_y = box[1] + ((box[3] - box[1])/2) # calculating center of the x side
      center_x = box[0] + ((box[2] - box[0])/2) # calculating center of the y side
      rect_y = center_y - img_size/2  # calculating starting x of rectangle
      rect_x = center_x - img_size/2  # calculating starting y of rectangle     
      
      # Cropping an area
      cropped_img = img[rect_y:rect_y+112,rect_x:rect_x+112]
      fname = './cropped/crop_test_' + filename    
      cv2.imwrite(fname, cropped_img)

      # Extract features
      cropped_img = model.get_input(cropped_img)
      features = model.get_features(cropped_img)
      comp.client_features = features
      # Compare features with existing ones
      res = comp.compare()
      if res > args.threshold:
        
        cv2.putText(img,text,(50,50), font, 1,(0,255,255),2,cv2.LINE_AA)

      cv2.rectangle(img, (rect_x, rect_y), (rect_x + 113, rect_y + 113), color, 2)  # rectangle around cropping area, we put 113 because sometimes rectangle also gets cropped
      # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)  # simple rectangle around face

      font = cv2.FONT_HERSHEY_SIMPLEX
      text = 'x: ' + str(box[0]) + ';  y: ' + str(box[1]) # + ' ' + str(box[2]) + ' ' + str(box[3])
      cv2.putText(img,text,(50,50), font, 1,(0,255,255),2,cv2.LINE_AA)
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
      # cv2.imwrite('./detections/' + filename, img)
      break
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
