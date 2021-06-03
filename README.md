# Real-Time Face Detection using OpenCV
> This project was written in Python
## What is OpenCV ?
OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library. The library has more than 2500 optimized algorithms, which includes a comprehensive set of both classic and state-of-the-art computer vision and machine learning algorithms. 

These algorithms can be used to detect and recognize faces, identify objects, classify human actions in videos, track camera movements, track moving objects, extract 3D models of objects, produce 3D point clouds from stereo cameras, stitch images together to produce a high resolution image of an entire scene, find similar images from an image database, remove red eyes from images taken using flash, follow eye movements, recognize scenery and establish markers to overlay it with augmented reality, etc.

## What is Face Detection ?
Face detection is a computer technology being used in a variety of applications that identifies human faces in digital images. The algorithms focus on the detection of frontal human faces. It is analogous to image detection in which the image of a person is matched bit by bit. Image matches with the image stores in database. Any facial feature changes in the database will invalidate the matching process.

```OpenCV``` contains many pre-trained classifiers for face, eyes, smile etc. The XML files of pre-trained classifiers are stored in ```opencv/data/```. For face detection specifically, there are two pre-trained classifiers:

1. Haar Cascade Classifier
2. LBP Cascade Classifier

We will explore only Haar Cascade Classifier here.

## What is Haar Cascade Classifier

It is a machine learning based approach where a cascade function is trained from a lot of positive (images with face) and negative images (images without face). The algorithm is proposed by Paul Viola and Michael Jones.

#### The algorithm has four stages:

1. ```Haar Feature Selection:``` Haar features are calculated in the subsections of the input image. The difference between the sum of pixel intensities of adjacent rectangular regions is calculated to differentiate the subsections of the image. A large number of haar-like features are required for getting facial features.

2. ```Creating an Integral Image:``` Too much computation will be done when operations are performed on all pixels, so an integral image is used that reduce the computation to only four pixels. This makes the algorithm quite fast.

3. ```Adaboost:``` All the computed features are not relevant for the classification purpose. Adaboost is used to classify the relevant features.

4. ```Cascading Classifiers:``` Now we can use the relevant features to classify a face from a non-face but algorithm provides another improvement using the concept of cascades of classifiers. Every region of the image is not a facial region so it is not useful to apply all the features on all the regions of the image. Instead of using all the features at a time, group the features into different stages of the classifier.Apply each stage one-by-one to find a facial region. If on any stage the classifier fails, that region will be discarded from further iterations. Only the facial region will pass all the stages of the classifier.

## How to use the Algorithm:
1. We first need to import some libraries.
```ruby
# Import libraries
import cv2
import numpy as np
```
2. We need to get the video capture from the webcam
```ruby
# Video capture using WebCam
cap = cv2.VideoCapture(0)  # The 0 means the first webcam that you have, if you have more webcam that you want to use you could put 1, 2, or 3... 
        
# print a feedback
print('Camera On')
```
3. We need to load the haar cascade clasifier (We could do this before step 2)
```ruby
# Load face detection classifier
# Load face detection classifier ~ Path to face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # Pre train model
```
4. We need to get frame by frame from the web cam capture
```ruby
while True:
  # Original frame ~ Video frame from camera
  ret, frame = cap.read()  # Return value (true or false) if the capture work, video frame
```
5. We need to convert original frame to a gray frame
```ruby
# Convert original frame to gray
gray = FrameEditing.convert_frame_to_gray(frame)
```
6. Get the face position
```
cascade.detectMultiScale(frame, scaleFactor, minNeighbors, minSize=Optional, maxSize=Optional)
   frame: have to be a gray frame bc most of the model are train with gray frame
   scaleFactor: parameter specifying how much the image size is reduced at each image scale. 1.05 is a good possible value for this, which means you use a small step for resizing, i.e. reduce size by 5%, you increase the chance of a matching size with the model for detection is found. (We have to scale the image bc the model has a fixed size defined during training).
   minNeighbors: parameter specifying how many neighbors each candidate rectangle should have to retain it. This parameter will affect the quality of the detected faces. Higher value results in less detections but with higher quality. 3~6 is a good value for it. (This is the quality for detecting faces)
   minSize: minimum possible object size. Objects smaller than that are ignored. This parameter determine how small size you want to detect. Usually, [30, 30] is a good start for face detection. (Optional ~ This is the min size we want to detect a face in a frame)
   maxSize: maximum possible object size. Objects bigger than this are ignored. This parameter determine how big size you want to detect. Usually, you don't need to set it manually, the default value assumes you want to detect without an upper limit on the size of the face. (Optional ~ This is the max size we want to detect a face in a frame)
```
```ruby
# Get location of the faces in term of position
# Return a rectangle (x_pos, y_pos, width, height)
faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)  
```
7. Draw a square around the face
```ruby
# Detect faces
for (x, y, w, h) in faces:
  # Draw rectangle in the face
  FrameDrawing.draw_rect(frame, (x, y), (x+w, y+h), (255, 53, 18), line_thickness)  # Rect for the face
```
8. We need to load the frames and show it into the screen
```ruby
# Load video frame
cv2.imshow('Video Frame', frame)
```
9. We need to quit getting frame after we press the key 'q'
```ruby
# Wait 1 millisecond second until q key is press
# Get a frame every 1 millisecond
if cv2.waitKey(1) == ord('q'):
  # Print feedback
  print('Camera Off')
  break
```
10. We need to realase the frames
```ruby
# Close windows
cap.release()  # Realise the webcam
cv2.destroyAllWindows()  # Destroy all the windows
```

## Final Code
```ruby
# Import libraries
import cv2
import numpy as np

# Video capture using WebCam
cap = cv2.VideoCapture(0)
        
# print a feedback
print('Camera On')

# Load face detection classifier
# Load face detection classifier ~ Path to face & eye cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # Pre train model

while True:
  # Original frame ~ Video frame from camera
  ret, frame = cap.read()  # Return value (true or false) if the capture work, video frame
  
  # Convert original frame to gray
  gray = FrameEditing.convert_frame_to_gray(frame)
  
  # Get location of the faces in term of position
  # Return a rectangle (x_pos, y_pos, width, height)
  faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE) 
  
  # Detect faces
  for (x, y, w, h) in faces:
     # Draw rectangle in the face
     FrameDrawing.draw_rect(frame, (x, y), (x+w, y+h), (255, 53, 18), line_thickness)  # Rect for the face
     
  # Load video frame
  cv2.imshow('Video Frame', frame)
  
  # Wait 1 millisecond second until q key is press
  # Get a frame every 1 millisecond
  if cv2.waitKey(1) == ord('q'):
     # Print feedback
     print('Camera Off')
     break
     
# Close windows
cap.release()  # Realise the webcam
cv2.destroyAllWindows()  # Destroy all the windows
```

## Note :pencil:
* Some of the function that you saw were created by me and you can find it in the documantation.
* If you don't have OpenCV install in your computer you can do it by typing ```pip install opencv-python``` or ```pip3 install opencv-python``` in your Terminal.
* If you don't have Numpy install in your computer you can do it by typing ```pip install numpy``` or ```pip3 install numpy``` in your Terminal.
* (I'm using a virtual environment [Anaconda](https://docs.anaconda.com/anaconda/install/) to install all the packages).
* This is one of the easiest and simplest ways to use OpenCV Face Detection. There are more things to learn about it so you can make the best with it. With this technology you can create amazing things.

## References
* [About OpenCV](https://opencv.org/about/)
* [Haar-cascade Detection](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)
* [Face Detection - Wikipedia](https://en.wikipedia.org/wiki/Face_detection)
* [OpenCV Classifier Better Explanation](https://github.com/informramiz/Face-Detection-OpenCV)
