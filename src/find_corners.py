import cv2
import numpy as np
import pandas as pd
import os
import sys

# Load the image
if(len(sys.argv) == 2):
    file_path = sys.argv[1]
else:
    file_path = 'input_imgs/image_pk.jpg'

image = cv2.imread(file_path)

r,c,temp = image.shape

# image = cv2.resize(image,(int(c/4),int(r/4)))

# Create a window to display the image
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', image)

# Create a list to store the annotated pixel positions
positions = []

# Select seven points with same sequence as given in report.pdf page 1

# Define a callback function to capture mouse events
def mouse_callback(event, x, y, flags, param):
    # Check if left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add the pixel position to the list
        positions.append([x, y, 1]) # type: ignore
        print('Annotated position:', (x, y))

# Set the mouse callback function for the image window
cv2.setMouseCallback('image', mouse_callback)

# Wait for the user to annotate the image
cv2.waitKey(0)

# Close the image window
cv2.destroyAllWindows()

xaxis11, xaxis12 = positions[0], positions[1]
xaxis21, xaxis22 = positions[2], positions[3]
xaxis31, xaxis32 = positions[4], positions[5]

yaxis11, yaxis12 = positions[0], positions[6]
yaxis21, yaxis22 = positions[2], positions[4]
yaxis31, yaxis32 = positions[3], positions[5]

zaxis11, zaxis12 = positions[0], positions[2]
zaxis21, zaxis22 = positions[1], positions[3]
zaxis31, zaxis32 = positions[6], positions[4]

###  X  ###
cv2.line(image,(xaxis11[0],xaxis11[1]),(xaxis12[0],xaxis12[1]),(0,0,255),2)        #blue
cv2.line(image,(xaxis21[0],xaxis21[1]),(xaxis22[0],xaxis22[1]),(0,0,255),2)
cv2.line(image,(xaxis31[0],xaxis31[1]),(xaxis32[0],xaxis32[1]),(0,0,255),2)

###  Y  ###
cv2.line(image,(yaxis11[0],yaxis11[1]),(yaxis12[0],yaxis12[1]),(0,255,0),2)     #green
cv2.line(image,(yaxis21[0],yaxis21[1]),(yaxis22[0],yaxis22[1]),(0,255,0),2)
cv2.line(image,(yaxis31[0],yaxis31[1]),(yaxis32[0],yaxis32[1]),(0,255,0),2)

###  Z  ###
cv2.line(image,(zaxis11[0],zaxis11[1]),(zaxis12[0],zaxis12[1]),(250,0,0),2)      #red
cv2.line(image,(zaxis21[0],zaxis21[1]),(zaxis22[0],zaxis22[1]),(250,0,0),2)
cv2.line(image,(zaxis31[0],zaxis31[1]),(zaxis32[0],zaxis32[1]),(255,0,0),2)

positions = np.array(positions)

df = pd.DataFrame({"X" : positions[:,0], "Y" : positions[:,1], "Z" : positions[:,2]})
df.to_csv("corner_coordinates.csv", index=False)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow("img", image)
cv2.waitKey(0)