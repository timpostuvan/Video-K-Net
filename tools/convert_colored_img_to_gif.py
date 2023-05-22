# Imports
from tools.utils.utils_visualization import cityscapes_cat2rgb
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
import argparse


# Declare a new path for the recolorized dataset:
def convert_to_gif(image_folder_name : str, output_path : str, segmentation : bool = True):
    # Define the name of the folder
    if segmentation:
        name = "segmentation"
    else:
        name = "instance"

    # Read images in the "segmentation" folder
    images = [img for img in os.listdir(os.path.join(PATH, name, image_folder_name))]
    
    # Define the frames
    frames = []
    for i in range(len(images)):
        frames.append(Image.open(os.path.join(PATH, name, image_folder_name, images[i])))

    # Save them as a gif with PIL
    frames[0].save(os.path.join(output_path, name, image_folder_name + '.gif'), format='GIF', append_images=frames[1:], save_all=True, duration=33, loop=0)



# Declare the main
if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Method to recolorize the segmentation and instance masks.")

    # Add an argument
    parser.add_argument("path", help="Path to the dataset you want to recolorize, see the README for more information.")

    # Parse the arguments
    PATH = parser.parse_args().path

    # Declare a new path for the recolorized dataset:
    NEW_PATH = os.path.join(PATH, "colorized_gif")

    # Set the path to the dataset you want to recolorize:
    PATH = os.path.join(PATH, "colorized")

    # Determine the number of jobs to run in parallel based on the minimum between the number of CPU cores and the number of the folders
    num_jobs = min(multiprocessing.cpu_count(), len(os.listdir(PATH)))

    # If the new path doesn't exist, create it
    if not os.path.exists(NEW_PATH):
        os.mkdir(NEW_PATH)
        print("Created the folder for the GIFs!")

    # Create the segmentation and instance folders
    if not os.path.exists(os.path.join(NEW_PATH, "segmentation")):
        os.mkdir(os.path.join(NEW_PATH, "segmentation"))
        print("Created the folder for the segmentation masks!")

    if not os.path.exists(os.path.join(NEW_PATH, "instance")):
        os.mkdir(os.path.join(NEW_PATH, "instance"))
        print("Created the folder for the instance masks!")

    print("Computing the GIFs in parallel")

    # Loop over all the files in the folder
    _ = Parallel(n_jobs=num_jobs)(delayed(convert_to_gif)(folder_name, NEW_PATH, segmentation=True) for folder_name in os.listdir(os.path.join(PATH, "segmentation")))
    _ = Parallel(n_jobs=num_jobs)(delayed(convert_to_gif)(folder_name, NEW_PATH, segmentation=False) for folder_name in os.listdir(os.path.join(PATH, "segmentation")))