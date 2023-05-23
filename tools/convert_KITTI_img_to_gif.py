# Imports
from visualize_initial_methods import cityscapes_cat2rgb
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
import argparse

####################################################
#                   TO MODIFY !
####################################################
# Path to the folder containing the images
PATH = r"C:\Users\Pierre\Desktop\training"

# Declare a new path for the recolorized dataset:
def convert_to_gif(image_folder_name : str, output_path : str):
    # Read images in the "segmentation" folder
    images = [img for img in os.listdir(os.path.join(PATH, image_folder_name))]
    
    # Define the frames
    frames = []
    for i in range(len(images)):
        frames.append(Image.open(os.path.join(PATH, image_folder_name, images[i])))

    # Save them as a gif with PIL
    frames[0].save(os.path.join(output_path, image_folder_name + '.gif'), format='GIF', append_images=frames[1:], save_all=True, duration=33, loop=0)



# Declare the main
if __name__ == "__main__":
    # Declare a new path for the recolorized dataset:
    NEW_PATH = os.path.join(PATH, "gif")

    # Set the path to the dataset you want to recolorize:
    PATH = os.path.join(PATH, "image_02")

    # Determine the number of jobs to run in parallel based on the minimum between the number of CPU cores and the number of the folders
    num_jobs = min(multiprocessing.cpu_count(), len(os.listdir(PATH)))

    # If the new path doesn't exist, create it
    if not os.path.exists(NEW_PATH):
        os.mkdir(NEW_PATH)
        print("Created the folder for the GIFs!")

    # Say that we do not have any bar for showing the progress
    print("Computing the GIFs in parallel, so no progress bar :(")

    # Loop over all the files in the folder
    _ = Parallel(n_jobs=num_jobs)(delayed(convert_to_gif)(folder_name, NEW_PATH) for folder_name in os.listdir(PATH))