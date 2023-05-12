#!/bin/sh
read -p "Enter image path:"
read img_path
echo "Image path is $img_path"
echo "Running python scripts find_corners.py"
echo "choose 7 corners in the image"
python src/find_corners.py "$img_path"
echo "Running python scripts svm.py"
echo "select top-left and bottom-right corners of the object"
python src/svm.py "$img_path"
echo "visulaizing 3d model"
view3dscene 3d_model.wrl
