# Import libraries
import os


####################################################
#                   TO MODIFY !
####################################################
# Path to the folder containing the images
PATH = r"C:\Users\Pierre\Desktop\video_knet_step-pretrained"

# Declare the main
if __name__ == "__main__":

    # cd to the folder containing this file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Print the beginning of the script
    print("Starting the colorization process...")

    # Execute colorize_img.py
    os.system(f'python colorize_img.py {PATH}')

    # Print the beginning of the script
    print("Starting the GIF creation process...")

    # Execute convert_colored_img_to_gif.py
    os.system(f'python convert_colored_img_to_gif.py {PATH}')