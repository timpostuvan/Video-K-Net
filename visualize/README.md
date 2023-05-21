# How to visualize the results
## Have the results in the correct format
Once the model is run, the validation results should be stored and have the following structure:

```
video_knet_step-pretrained
├───final
│   ├─── ...
└───panoptic
    ├─── ...
```

## Modify the PATH variable in [visualize.py](visualize.py) and run it
The PATH variable should point to the folder containing the results. Running the [visualize.py](visualize.py) script will then create the folders `colorized` and `colorized_gif` in the same folder as the results. The `colorized` folder will contain the colorized images. The `colorized_gif` folder will contain the colorized images in gif format.

In the end, the results folder structure should look like this:

```
video_knet_step-pretrained
├───colorized
│   ├───instance
│   │   ├─── ...
│   └───segmentation
│       ├─── ...
├───colorized_gif
│   ├───instance
│   └───segmentation
├───final
│   ├─── ...
└───panoptic
    ├─── ...
```

## Example
The results folder is stored in `C:\Users\Pierre\Desktop\video_knet_step-pretrained`.

The PATH variable in `visualize.py` should then be set to `C:\Users\Pierre\Desktop\video_knet_step-pretrained`.

If the results folder has the correct structure, the GIFs should be obtained under `C:\Users\Pierre\Desktop\video_knet_step-pretrained\colorized_gif`.