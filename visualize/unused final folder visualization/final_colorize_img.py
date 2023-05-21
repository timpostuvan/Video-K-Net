# Imports
from visualizer import cityscapes_cat2rgb, trackmap2rgb
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

# Set the path to the dataset you want to recolorize:
PATH = r"C:\Users\Pierre\Desktop\video_knet_step-pretrained\final"

# Declare a new path for the recolorized dataset:
NEW_PATH = r"C:\Users\Pierre\Desktop\video_knet_step-pretrained\colorized"

# Determine the number of jobs to run in parallel based on the number of CPU cores
num_jobs = multiprocessing.cpu_count()

# If the new path doesn't exist, create it
if not os.path.exists(NEW_PATH):
    os.mkdir(NEW_PATH)
    print("Created the folder for the recolorized dataset!")

# Method to get transform the images in a folder
def save_recolorized_images(folder : str, filename : str):
    """ 
    Transforms an image to its recolorized version. Takes the folder and filename as input.
    """

    # Open the image in grayscale
    img = Image.open(os.path.join(PATH, folder, filename)).convert("L")

    # Convert the image to a numpy array
    img = np.array(img)

    # Recolorize the image
    img = trackmap2rgb(img)

    # Save the image
    cv2.imwrite(os.path.join(NEW_PATH, folder, filename), img)

# Loop over all the folders in PATH
for folder in tqdm(os.listdir(PATH)):
    # If the folder doesn't exist, create it
    if not os.path.exists(os.path.join(NEW_PATH, folder)):
        os.mkdir(os.path.join(NEW_PATH, folder))

    # Loop over all the files in the folder
    _ = Parallel(n_jobs=num_jobs)(delayed(save_recolorized_images)(folder, filename) for filename in os.listdir(os.path.join(PATH, folder)))

