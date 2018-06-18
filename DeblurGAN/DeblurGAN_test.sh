#!/bin/bash
echo "begin"
for file in $(ls blur_photo);do
echo ${file%.*}
python deblur_image.py --image_path=blur_photo/${file} 
done
echo "end"
