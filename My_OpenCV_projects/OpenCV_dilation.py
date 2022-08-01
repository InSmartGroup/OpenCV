import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create an empty matrix
im = np.zeros((10, 10), dtype='uint8')
plt.imshow(im)

# Adding some white blobs to the matrix
im[0, 1] = 1
im[4, 0] = 1
im[6, -1] = 1
im[3, 3] = 1
im[5:7, 5:7] = 1
im[6:8, 1:3] = 1
im[1:3, 8:10] = 1
im[-2, 7] = 1
im[-1, 4] = 1
im[1, 5] = 1
plt.imshow(im)

# Create an ellipse structuring element
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
plt.imshow(kernel)

# Specify a few variables for future manual dilation operation
kernel_size = kernel.shape[0]
im_height, im_width = im.shape[:]
border = kernel_size // 2

# Check the cv2 dilation operation
opencv_dilated_image = cv2.dilate(im, kernel)
plt.imshow(opencv_dilated_image)

# Write the dilation operation from scratch
padded_im = np.zeros((im_height + border * 2, im_width + border * 2))
padded_im = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value=0)

fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
video = cv2.VideoWriter('Dilation.mp4', fourcc, 10, (50, 50))

for h_i in range(border, im_height + border):
    for w_i in range(border, im_width + border):
        if im[h_i - border, w_i - border]:
            padded_im[h_i - border:(h_i + border) + 1, w_i - border:(w_i + border) + 1] = \
                cv2.bitwise_or(padded_im[h_i - border:(h_i + border) + 1, \
                               w_i - border:(w_i + border) + 1], kernel)

            resized_frame = cv2.resize(padded_im, (50, 50))
            resized_frame = resized_frame * 255
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BAYER_GB2BGR)
            video.write(resized_frame)

video.release()

# Check if there's a difference between cv2.dilate and a manually created dilation
dilated_image = padded_im[border:border + im_height, border:border + im_width]

plt.figure(figsize=[10, 10])
plt.subplot(121); plt.title('OpenCV dilation using looping'); plt.imshow(opencv_dilated_image)
plt.subplot(122); plt.title('Manual dilation'); plt.imshow(dilated_image)
