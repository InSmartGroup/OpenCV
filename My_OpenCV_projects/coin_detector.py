import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Read an image in color
source_image = cv2.imread('CoinsA.png', 1)
print(source_image.shape)

source_image_copy = source_image.copy()

plt.imshow(source_image[:, :, ::-1])

# Convert an image to grayscale
image_grayscale = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
print(image_grayscale.shape)

plt.subplot(121); plt.title('Original image'); plt.imshow(source_image[:, :, ::-1])
plt.subplot(122); plt.title('Grayscale image'); plt.imshow(image_grayscale)

# Split an image to separate color channels
blue_channel, green_channel, red_channel = cv2.split(source_image)

plt.figure(figsize=[20, 15])
plt.subplot(141); plt.title('Original image'); plt.imshow(source_image[:, :, ::-1])
plt.subplot(142); plt.title('Blue channel'); plt.imshow(blue_channel)
plt.subplot(143); plt.title('Green channel'); plt.imshow(green_channel)
plt.subplot(144); plt.title('Red channel'); plt.imshow(red_channel)

# Threshold an image
# Need to play with the minimum threshold to figure out the best quality with minimum white inner blobs
retval, image_thresh = cv2.threshold(green_channel, 20, 255, cv2.THRESH_BINARY_INV)
print(image_thresh.shape)

plt.imshow(image_thresh)

# Perform morphological operations
# Create a structuring element first
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

# Morph an image
image_morphed = cv2.morphologyEx(image_thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

plt.imshow(image_morphed)

# Create a blob detector and detect coins
# Create the blob detector parameters first
blob_detector_params = cv2.SimpleBlobDetector_Params()

blob_detector_params.blobColor = 0
blob_detector_params.minDistBetweenBlobs = 2

blob_detector_params.filterByArea = False

blob_detector_params.filterByCircularity = True
blob_detector_params.minCircularity = 0.8

blob_detector_params.filterByConvexity = True
blob_detector_params.minConvexity = 0.8

blob_detector_params.filterByInertia = True
blob_detector_params.minInertiaRatio = 0.8

# Create the blob detector and apply the parameters
blob_detector = cv2.SimpleBlobDetector_create(blob_detector_params)

# Detect the blobs
blob_detector_keypoints = blob_detector.detect(image_morphed)
print(f"The number of blobs: {len(blob_detector_keypoints)}")

# Draw circles around detected blobs
for key in blob_detector_keypoints:
    # Identify the x_axis (width) and the y_axis (height)
    width, height = key.pt
    width = int(round(width))
    height = int(round(height))
    cv2.circle(source_image_copy, (width, height), 3, (255, 255, 255), -1)
    
    # Identify the diameter and radius of an outer circle
    diameter = key.size
    radius = int(round(diameter / 2))
    cv2.circle(source_image_copy, (width, height), radius, (0, 255, 255), 2)
    
plt.imshow(source_image_copy[:, :, ::-1])
