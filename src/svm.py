import numpy as np
import cv2
import pandas as pd
import os
import sys
from scipy.linalg import cholesky

BASE_DIR = os.getcwd()

positions = []
# Define a callback function to capture mouse events
def mouse_callback(event, x, y, flags, param):
    # Check if left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add the pixel position to the list
        positions.append([x, y, 1])
        print('Annotated position:', (x, y))

# Load the image
if(len(sys.argv) == 2):
    file_path = sys.argv[1]
else:
    file_path = 'input_imgs/image1.jpg'

img = cv2.imread(file_path)

r,c,temp = img.shape
# img = cv2.resize(img,(int(c/4),int(r/4)))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

df = pd.read_csv('corner_coordinates.csv')
data = np.array(df)

xaxis11, xaxis12 = data[0], data[1]
xaxis21, xaxis22 = data[2], data[3]
xaxis31, xaxis32 = data[4], data[5]

yaxis11, yaxis12 = data[0], data[6]
yaxis21, yaxis22 = data[2], data[4]
yaxis31, yaxis32 = data[3], data[5]

zaxis11, zaxis12 = data[0], data[2]
zaxis21, zaxis22 = data[1], data[3]
zaxis31, zaxis32 = data[6], data[4]

world_origin = data[0]
reference_point_x, reference_point_y, reference_point_z = data[1], data[6], data[2]

world_origin = np.array([world_origin])
reference_point_x = np.array([reference_point_x])
reference_point_y = np.array([reference_point_y])
reference_point_z = np.array([reference_point_z])


ax1,bx1,cx1 = np.cross(xaxis11,xaxis12)
ax2,bx2,cx2 = np.cross(xaxis21,xaxis22)
vanishing_point_x = np.cross([ax1,bx1,cx1],[ax2,bx2,cx2])
vanishing_point_x = vanishing_point_x / vanishing_point_x[2]

ay1,by1,cy1 = np.cross(yaxis11,yaxis12)
ay2,by2,cy2 = np.cross(yaxis21,yaxis22)
vanishing_point_y = np.cross([ay1,by1,cy1],[ay2,by2,cy2])
vanishing_point_y = vanishing_point_y / vanishing_point_y[2]

az1,bz1,cz1 = np.cross(zaxis11,zaxis12)
az2,bz2,cz2 = np.cross(zaxis21,zaxis22)
vanishing_point_z = np.cross([az1,bz1,cz1],[az2,bz2,cz2])
vanishing_point_z = vanishing_point_z / vanishing_point_z[2]

print(f"vanishing point along the x axis: {vanishing_point_x}")
print(f"vanishing point along the y axis: {vanishing_point_y}")
print(f"vanishing point along the z axis: {vanishing_point_z}")

length_x = np.sqrt(np.sum(np.square(reference_point_x - world_origin)))   
length_y = np.sqrt(np.sum(np.square(reference_point_y - world_origin)))   
length_z = np.sqrt(np.sum(np.square(reference_point_z - world_origin)))   


print("finding the Kronecker product")
Kxy = np.kron(vanishing_point_x, vanishing_point_y)
Kyz = np.kron(vanishing_point_y, vanishing_point_z)
Kzx = np.kron(vanishing_point_z, vanishing_point_x)

kron_matrix = np.array([[Kxy[0] + Kxy[4], Kxy[2] + Kxy[6], Kxy[5] + Kxy[7]],
               [Kyz[0] + Kyz[4], Kyz[2] + Kyz[6], Kyz[5] + Kyz[7]],
                [Kzx[0] + Kzx[4], Kzx[2] + Kzx[6], Kzx[5] + Kzx[7]]])

# print(kron_matrix.shape)
b = np.array([-1, -1, -1])

omega = np.linalg.solve(kron_matrix, b)

# print(omega)

omega_matrix = np.array([[omega[0], 0, omega[1]],
                         [0, omega[0], omega[2]],
                         [omega[1], omega[2], 1]]) # type: ignore

print("omega matrix:")
print(omega_matrix)

intrinsic_matrix = np.linalg.inv(cholesky(omega_matrix))

print("camera intrinsic matrix:")
print(intrinsic_matrix)

ax,resid,rank,s = np.linalg.lstsq( (vanishing_point_x-reference_point_x).T , (reference_point_x - world_origin).T )
ax = ax[0][0]/length_x

ay,resid,rank,s = np.linalg.lstsq( (vanishing_point_y-reference_point_y).T , (reference_point_y - world_origin).T )
ay = ay[0][0]/length_y

az,resid,rank,s = np.linalg.lstsq( (vanishing_point_z-reference_point_z).T , (reference_point_z - world_origin).T )
az = az[0][0]/length_z

px, py, pz = ax*vanishing_point_x, ay*vanishing_point_y, az*vanishing_point_z

homographic_mat = np.zeros((3, 4))

homographic_mat[:, 0], homographic_mat[:, 1], homographic_mat[:, 2], homographic_mat[:, 3] = px, py, pz, world_origin

print("Homography matrix:")
print(homographic_mat)

homographic_mat_xy, homographic_mat_yz, homographic_mat_zx = np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))

homographic_mat_xy[:,0], homographic_mat_xy[:,1], homographic_mat_xy[:,2] = px, py, world_origin
homographic_mat_yz[:,0], homographic_mat_yz[:,1], homographic_mat_yz[:,2] = py, pz, world_origin
homographic_mat_zx[:,0], homographic_mat_zx[:,1], homographic_mat_zx[:,2] = px, pz, world_origin

homographic_mat_xy[0,2] = homographic_mat_xy[0,2] 
homographic_mat_xy[1,2] = homographic_mat_xy[1,2]

homographic_mat_yz[0,2] = homographic_mat_yz[0,2]
homographic_mat_yz[1,2] = homographic_mat_yz[1,2] 

homographic_mat_zx[0,2] = homographic_mat_zx[0,2]
homographic_mat_zx[1,2] = homographic_mat_zx[1,2]

r,c,temp = img.shape
Transformed_image_xy = cv2.warpPerspective(img,homographic_mat_xy,(r,c),flags=cv2.WARP_INVERSE_MAP)      # homographic transformation in xy plane
Transformed_image_yz = cv2.warpPerspective(img,homographic_mat_yz,(r,c),flags=cv2.WARP_INVERSE_MAP)      # homographic transformation in yz plane
Transformed_image_zx = cv2.warpPerspective(img,homographic_mat_zx,(r,c),flags=cv2.WARP_INVERSE_MAP)      # homographic transformation in zx plane

cv2.namedWindow('Transformed_image_xy', cv2.WINDOW_NORMAL)
cv2.imshow("Transformed_image_xy",Transformed_image_xy)
# select two points. Top left and bottom right that to be cropped.
cv2.setMouseCallback('Transformed_image_xy', mouse_callback)
cv2.waitKey(0)
x1, y1, x2, y2 = positions[0][0], positions[0][1], positions[1][0], positions[1][1]
cropped_XY = Transformed_image_xy[y1:y2, x1:x2]

cv2.imwrite(os.path.join(BASE_DIR, "texture_imgs/XY.jpg"),Transformed_image_xy)
cv2.imwrite(os.path.join(BASE_DIR, "texture_imgs/XY_cropped.jpg"), cropped_XY)
cv2.destroyAllWindows()
print("XY projection image saved")

positions = []
cv2.namedWindow('Transformed_image_yz', cv2.WINDOW_NORMAL)
cv2.imshow("Transformed_image_yz",Transformed_image_yz)
cv2.setMouseCallback('Transformed_image_yz', mouse_callback)
cv2.waitKey(0)
x1, y1, x2, y2 = positions[0][0], positions[0][1], positions[1][0], positions[1][1]
cropped_YZ = Transformed_image_yz[y1:y2, x1:x2]
cv2.imwrite(os.path.join(BASE_DIR, "texture_imgs/YZ.jpg"),Transformed_image_yz)
cv2.imwrite(os.path.join(BASE_DIR, "texture_imgs/YZ_cropped.jpg"), cropped_YZ)
cv2.destroyAllWindows()
print("YZ projection image saved")

positions = []
cv2.namedWindow('Transformed_image_zx', cv2.WINDOW_NORMAL)
cv2.imshow("Transformed_image_zx",Transformed_image_zx)
cv2.setMouseCallback('Transformed_image_zx', mouse_callback)
cv2.waitKey(0)
x1, y1, x2, y2 = positions[0][0], positions[0][1], positions[1][0], positions[1][1]
cropped_ZX = Transformed_image_zx[y1:y2, x1:x2]
cv2.imwrite(os.path.join(BASE_DIR, "texture_imgs/ZX.jpg"),Transformed_image_zx)
cv2.imwrite(os.path.join(BASE_DIR, "texture_imgs/ZX_cropped.jpg"), cropped_ZX)
cv2.destroyAllWindows()
print("ZX projection image saved")
