# Imports
from tools.utils.utils_visualization import cityscapes_cat2rgb, trackmap2rgb
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
import argparse


# Method to get transform the images in a folder
def save_recolorized_images(folder : str, filename : str, segmentation : bool = True):
    """ 
    Transforms an image to its recolorized version. Takes the folder and filename as input.
    """

    # Open the image in grayscale
    img = Image.open(os.path.join(PATH, folder, filename)).convert("L")

    # Convert the image to a numpy array
    img = np.array(img)

    # Recolorize the image
    img = trackmap2rgb(img)

    # Save the image under the correct folder
    if segmentation:
        save_path = os.path.join(NEW_PATH, "segmentation", folder, filename)
    else:
        save_path = os.path.join(NEW_PATH, "instance", folder, filename)

    # Save the image
    cv2.imwrite(save_path, img)



# Declare the main
if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Method to recolorize the segmentation and instance masks.")

    # Add an argument
    parser.add_argument("path", help="Path to the dataset you want to recolorize, see the README for more information.")

    # Parse the arguments
    PATH = parser.parse_args().path

    # Declare a new path for the recolorized dataset:
    NEW_PATH = os.path.join(PATH, "colorized")

    # Set the path to the dataset you want to recolorize:
    PATH = os.path.join(PATH, "panoptic")

    # Determine the number of jobs to run in parallel based on the number of CPU cores
    num_jobs = multiprocessing.cpu_count()

    # If the new path doesn't exist, create it
    if not os.path.exists(NEW_PATH):
        os.mkdir(NEW_PATH)
        print("Created the folder for the recolorized dataset!")

    # if the folders "segmentation" and "instance" don't exist in NEW_PATH, create them
    if not os.path.exists(os.path.join(NEW_PATH, "segmentation")):
        os.mkdir(os.path.join(NEW_PATH, "segmentation"))
        print("Created the folder for the segmentation masks!")

    if not os.path.exists(os.path.join(NEW_PATH, "instance")):
        os.mkdir(os.path.join(NEW_PATH, "instance"))
        print("Created the folder for the instance masks!")

    # Loop over all the folders in PATH
    for folder in tqdm(os.listdir(PATH)):
        # If the folder doesn't exist, create it
        if not os.path.exists(os.path.join(NEW_PATH, "segmentation", folder)):
            os.mkdir(os.path.join(NEW_PATH, "segmentation", folder))
        
        if not os.path.exists(os.path.join(NEW_PATH, "instance", folder)):
            os.mkdir(os.path.join(NEW_PATH, "instance", folder))

        # Loop over all the files in the folder that contain "cat" in their name
        _ = Parallel(n_jobs=num_jobs//2)(delayed(save_recolorized_images)(folder, filename, segmentation=True) for filename in os.listdir(os.path.join(PATH, folder)) if "cat" in filename)

        # Loop over all the files in the folder that contain "ins" in their name
        _ = Parallel(n_jobs=num_jobs//2)(delayed(save_recolorized_images)(folder, filename, segmentation=False) for filename in os.listdir(os.path.join(PATH, folder)) if "ins" in filename)
