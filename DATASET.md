Please prepare the datasets according to the following instructions.

The final dataset folder should look like this: 
```
root 
├── data
│   ├──  kitti-step
│   ├──  cityscapes
```


### KITTI-STEP Dataset

KITTI-STEP benchmark consists of 21 training sequences and 29 test sequences, where each sequence has its corresponding video panoptic segmentation masks: semantic segmentation masks and instance segmentation masks. The benchmark is based on the KITTI Tracking Evaluation and the Multi-Object Tracking and Segmentation (MOTS) benchmark.

To prepare KITTI-STEP dataset, run the following script that downloads the dataset and preprocesses it in the required format:

```bash
bash ./tools/download_and_preprocess_KITTI_STEP.sh
```

The preprocessed dataset should have the following structure:

```
├── kitti-step
│   ├──  video_sequence
│   │   ├── train
            ├──00018_000331_leftImg8bit.png
            ├──000018_000331_panoptic.png
            ├──****
│   │   ├── val
│   │   ├── test 
```

You can get already preprocessed dataset at https://huggingface.co/LXT/VideoK-Net/tree/main.


### Cityscapes Dataset

Cityscapes dataset is a high-resolution road-scene dataset which contains 19 classes (8 thing classes and 11 stuff classes). It contains 2975 images for training, 500 images for validation, and 1525 images for testing, where each image has also its corresponding panoptic segmentation mask: semantic segmentation mask and instance segmentation mask.

Preparing Cityscapes dataset has four steps:

1. Download the Cityscapes dataset from the official website (images: ``leftImg8bit_trainvaltest.zip``, annotations: ``gtFine_trainvaltest.zip``).

2. Convert annotations into required format using the official scripts [Github repo](https://github.com/mcordts/cityscapesScripts). When running the following commands, set ``CITYSCAPES_DATASET`` environmental variable to the location of annotations.

```bash
# install package for preprocessing Cityscapes
git clone https://github.com/mcordts/cityscapesScripts
cd cityscapesScripts
python -m pip install cityscapesscripts

# convert annotations into png images with instance IDs
python cityscapesscripts/preparation/createTrainIdInstanceImgs.py

# convert annotations into png images with label IDs
python cityscapesscripts/preparation/createTrainIdLabelImgs.py 

# convert annotations into COCO panoptic segmentation format
python cityscapesscripts/preparation/createPanopticImgs.py
```

3. Download COCO-like annotations (``instancesonly_filtered_gtFine_*.json``) that can be used for instance segmentation training from [here](https://onedrive.live.com/?authkey=%21AJNRLRBY2KbbMbE&id=4155EADDA5C5262E%21169&cid=4155EADDA5C5262E). Then, put them into ``annotations`` folder, together with generated ``cityscapes_panoptic_*.json``.

4. Move ``license.txt`` to ``cityscapes`` folder.

The final folder should look like this:

```
├── cityscapes
│   ├── annotations
│   │   ├── instancesonly_filtered_gtFine_train.json # coco instance annotation file (COCO format)
│   │   ├── instancesonly_filtered_gtFine_val.json
│   │   ├── cityscapes_panoptic_train.json  # panoptic json file 
│   │   ├── cityscapes_panoptic_val.json  
│   ├── leftImg8bit
│   ├── gtFine
│   │   ├──cityscapes_panoptic_{train,val}/  # png annotations
│   │   
│   ├── license.txt
```
