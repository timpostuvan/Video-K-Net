#!/bin/bash

# 0. cd to the root folder
cd ..

# 1. Make a folder for the KITTI dataset
mkdir -p ./data/

# 2. Install the KITTI dataset
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip -O ./data/imgs.zip
wget https://storage.googleapis.com/gresearch/tf-deeplab/data/kitti-step.tar.gz -O ./data/masks.tar.gz

# 3. Unzip the dataset and the masks
unzip ./data/imgs.zip
tar -xzf ./data/masks.tar.gz

# 4. Remove the zip files
rm ./data/imgs.zip
rm ./data/masks.tar.gz

# 5. Move the training, testing and kitti-step folders to the data folder
mv ./training ./data/training
mv ./testing ./data/testing
mv ./kitti-step ./data/kitti-step

# 6. Create the folder panoptic under the training folder
mkdir -p ./data/training/panoptic

# 7. Move all the interesting sub-folder in kitti-step to the panoptic folder
mv ./data/kitti-step/*/*/* ./data/training/panoptic

# 8. Remove the kitti-step folder
rm -rf ./data/kitti-step
mkdir -p ./data/kitti-step

# 9. Apply scripts/kitti_step_prepare.py
python3 ./scripts/kitti_step_prepare.py

# 10. Create a results folder
mkdir -p ./results