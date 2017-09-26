
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('bbox-example-image.jpg')

# Define a function that takes an image, a list of bounding boxes, 
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn
    return draw_img # Change this line to return image copy with boxes
# Add bounding boxes in this format, these are just example coordinates.
bboxes = [((100, 100), (200, 200)), ((300, 300), (400, 400))]

result = draw_boxes(image, bboxes)
plt.imshow(result)

#
# code & comments in this file taken from lesson 5. Manual Vehicle Detection
#

# Define a function that takes an image, a list of bounding boxes, 
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn

# You'll draw bounding boxes with cv2.rectangle() like this:

cv2.rectangle(image_to_draw_on, (x1, y1), (x2, y2), color, thick)

# In this call to cv2.rectangle() your image_to_draw_on should be the copy of your image,
# then (x1, y1) and (x2, y2) are the x and y coordinates of any two opposing corners 
# of the bounding box you want to draw. color is a 3-tuple, for example, (0, 0, 255) for blue, 
# [given an RGB image] and thick is an optional integer parameter to define the box thickness. 