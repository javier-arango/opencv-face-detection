# Real-Time Face Detection using OpenCV
> This project was written in Python
## What is OpenCV ?
OpenCV (Open Source Computer Vision Library) is a free and open source software library for computer vision and machine learning. More than 2500 optimized algorithms are included in the library, which contains a comprehensive mix of both classic and cutting-edge computer vision and machine learning techniques. 

These algorithms can be used to detect and recognize faces, identify objects, classify human actions in videos, track camera movements, track moving objects, extract 3D models of objects, produce 3D point clouds from stereo cameras, stitch images together to produce a high resolution image of an entire scene, find similar images from an image database, remove red eyes from images taken using flash, follow eye movements, recognize scenery and establish markers to overlay it with augmented reality, etc.

## What is Face Detection ?
Face detection is a computer technique that recognizes human faces in digital images and is used in a range of applications. The algorithms are designed to recognize frontal human faces. It's similar to image detection, where a person's image is matched bit by bit. The image is identical to the image stored in the database. Any changes to the database's facial features will render the matching process ineffective.

```OpenCV``` contains many pre-trained classifiers for face, eyes, smile etc. The XML files of pre-trained classifiers are stored in ```opencv/data/```. For face detection specifically, there are two pre-trained classifiers:

1. Haar Cascade Classifier
2. LBP Cascade Classifier

We will explore only Haar Cascade Classifier here.

## What is the Haar Cascade Classifier ?

It's a machine-learning-based methodology in which a cascade function is learned using a large number of positive (face-based) and negative (non-face-based) images (images without face). Paul Viola and Michael Jones proposed the algorithm.

#### The algorithm has four stages:

1. ```Haar Feature Selection:``` Haar features are calculated in the input image's subsections. To distinguish the image's subsections, the difference between the sum of pixel intensities of adjacent rectangular regions is calculated. Obtaining facial features necessitates a large number of haar-like features.

2. ```Creating an Integral Image:``` Too much computation will be done when operations are performed on all pixels, so an integral image is used that reduce the computation to only four pixels. This makes the algorithm quite fast.

3. ```Adaboost:``` All the computed features are not relevant for the classification purpose. Adaboost is used to classify the relevant features.

4. ```Cascading Classifiers:``` We can now classify a face from a non-face using relevant information, but the technique gives additional enhancement by utilizing the concept of cascades of classifiers. Because every region of the image is not a facial region, applying all of the features to every region of the image is pointless. Rather than using all of the features at once, divide them into distinct phases of the classifier. To find a facial region, apply each stage one at a time. If the classifier fails at any point throughout the process, that region will be skipped over in subsequent iterations. Only the facial region will pass all of the classifier's stages.

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
3. We need to load the haar cascade classifier (We could do this before step 2)
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
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```
6. Get the face position
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
  cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 53, 18), 2)  # Rect for the face
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
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
  # Get location of the faces in term of position
  # Return a rectangle (x_pos, y_pos, width, height)
  faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE) 
  
  # Detect faces
  for (x, y, w, h) in faces:
     # Draw rectangle in the face
     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 53, 18), 2)  # Rect for the face
     
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
* In this project, I added some things differently like combining four frames into one, show only the skin color, draw the coordinates and size into the screen, and more.
* If you don't have OpenCV install in your computer you can do it by typing ```pip install opencv-python``` or ```pip3 install opencv-python``` in your Terminal.
* If you don't have Numpy install in your computer you can do it by typing ```pip install numpy``` or ```pip3 install numpy``` in your Terminal.
* (I'm using a virtual environment [Anaconda](https://docs.anaconda.com/anaconda/install/) to install all the packages).
* The ```config``` folder contain methods to edit the video frame and to draw into the frame.
* This is one of the easiest and simplest ways to use OpenCV Face Detection. There are more things to learn about it so you can make the best with it. With this technology you can create amazing things.

## References
* [About OpenCV](https://opencv.org/about/)
* [Haar-cascade Detection](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)
* [Face Detection - Wikipedia](https://en.wikipedia.org/wiki/Face_detection)
* [OpenCV Classifier Better Explanation](https://github.com/informramiz/Face-Detection-OpenCV)
