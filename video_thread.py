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
from retinaface import RetinaFace
import skimage
import threading
from shutil import copyfile

# Reading folder changes and performing face detection and cropping
def readFolder():
	path_to_watch = '/home/ti/Downloads/DATA/first'
	where_to_move = '/home/ti/Downloads/DATA/second'
	before = dict([(f, None) for f in os.listdir(path_to_watch)])
	while 1:
	  time.sleep(1)
	  after = dict([(f, None) for f in os.listdir(path_to_watch)])
	  added = [f for f in after if not f in before]
	  if added > 0: 
	  	for f in added:
	  		copyfile(path_to_watch + '/' + f, where_to_move + '/' + f)

	  # print(added)
	  removed = [f for f in before if not f in after]
	  if added: print "Added: ", ", ".join(added)
	  if removed: print "Removed: ", ", ".join(removed)
	  before = after


count = 1

gpuid = 0
# Loading model for face detection
detector = RetinaFace('/home/ti/Downloads/MODELS/retinaface-R50/R50', 0, gpuid, 'net3')
# Reading stream from camera
# fvs = WebcamVideoStream(src='rtsp://admin:Admin123!@10.150.30.202:554/live').start()	# back of the office
fvs = WebcamVideoStream(src='rtsp://admin:Admin1234!@10.150.30.203:554/live').start()	# inside the office
time.sleep(1.0)

# Start fps counter
fps = FPS().start()

# Start folder reading in thread
thread = threading.Thread(target=readFolder, args=())
thread.daemon = True
thread.start() 

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
      color = (0,255,0)
      print(box)

      filename = str(datetime.datetime.now()).replace(":", "_").replace(".", "_").replace("-", "_").replace(' ', '_') + '.jpg'
      # Calculate cropping area
      img_size = 112
      center_y = box[1] + ((box[3] - box[1])/2) 	# calculating center of the x side
      center_x = box[0] + ((box[2] - box[0])/2) 	# calculating center of the y side
      rect_y = center_y - img_size/2  				# calculating starting x of rectangle
      rect_x = center_x - img_size/2  				# calculating starting y of rectangle
      
      cv2.rectangle(img, (rect_x, rect_y), (rect_x + 114, rect_y + 114), color, 2)  	# rectangle around cropping area, we put 114 because sometimes borders of rectangle also get cropped
      # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)  # simple rectangle around face

      font = cv2.FONT_HERSHEY_SIMPLEX
      text = 'x: ' + str(box[0]) + ';  y: ' + str(box[1]) # + ' ' + str(box[2]) + ' ' + str(box[3])
      cv2.putText(img,text,(50,50), font, 1, (0,255,255), 2, cv2.LINE_AA)
      
      print('Faces found', filename)
      cv2.imwrite('/home/ti/Downloads/DATA/first/' + filename, img)
      # break
                
    # cv2.imshow('image', img)      
        
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
