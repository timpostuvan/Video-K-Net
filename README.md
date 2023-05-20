# Ablation Studies of Video K-Net: A Simple, Strong, and Unified Baseline for Video Segmentation

We explore modifications of Video K-Net, a simple, strong, and unified framework for fully end-to-end dense video segmentation. Video K-Net is built upon K-Net, a method of unifying image segmentation via a group of learnable kernels.

This project contains the training and evaluation code of Video K-Net for VPS (Video Panoptic Segmentation) on KITTI-STEP dataset.


### Environment and Dataset Preparation 
The codebase is based on MMDetection and MMSegmentation. Parts of the code are borrowed from UniTrack.
* Nvidia device with CUDA 
* Python 3.7+
* PyTorch 1.7.0+
* torchvision 0.8.0+
* MIM >= 0.1.1
* MMCV-full >= v1.3.8
* MMDetection == v2.18.0
* Other python packages in requirements.txt

#### (Recommended) Install with Conda

Install conda from [here](https://repo.anaconda.com/miniconda/), Miniconda3-latest-(OS)-(platform).
```shell
# 1. Create a conda virtual environment.
conda create -n video-k-net python=3.7 -y
conda activate video-k-net

# 2. Install PyTorch
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# 3. Install MMCV and MMDetection
pip install openmim
mim install mmcv-full==1.3.14
mim install mmdet==2.18.0

# 4. Install other dependencies
pip install -r requirements.txt
```

For instructions how to obtain and properly prepare the datasets, see the [DATASET.md](https://github.com/timpostuvan/Video-K-Net/blob/main/DATASET.md).


### Checkpoints of Pretrained K-Net Models and Trained Video K-Net Models

We provide checkpoints of pretrained and trained models. The pretrained K-Net models can be used as an initialization to train the Video K-Net, while the trained Video K-Net models are ready for play and test.

**TODO**: Google Drive Link: [here]()


### Training on KITTI-STEP

1. Pretrain K-Net on Cityscapes-STEP dataset. This step is very important to improve the segmentation performance.

Cityscape-STEP follows the format of STEP: 17 stuff classes and 2 thing classes. 

```bash
# train Cityscapes STEP panoptic segmentation model on 3 GPUs
bash ./tools/dist_train.sh configs/knet_cityscapes_step/knet_s3_r50_fpn.py 3 $WORK_DIR --no-validate
```

2. Train the Video K-Net on KITTI-STEP. We have provided checkpoints of the pretrained K-Net models on Cityscapes-STEP.

```bash
# train Video K-Net on KITTI-step with 3 GPUs from pretrained checkpoint
bash ./tools/dist_train.sh configs/video_knet_kitti_step/video_knet_s3_r50_rpn_1x_kitti_step_sigmoid_stride2_mask_embed_link_ffn_joint_train.py 3 $WORK_DIR --no-validate --load-from $CHECKPOINT
```

It is also possible to train Video K-Net from scratch, however, this results in significantly inferior performance.

```bash
# train Video K-Net on KITTI-step with 3 GPUs from scratch
bash ./tools/dist_train.sh configs/video_knet_kitti_step/video_knet_s3_r50_rpn_1x_kitti_step_sigmoid_stride2_mask_embed_link_ffn_joint_train.py 3 $WORK_DIR --no-validate
```

The above commands use the original Video K-Net architecture. To train a modified architecture or in a different experimental setting, change the configuration file.


### Evaluation on KITTI-STEP

1. Generate predictions on validation set.

```bash
# generate predictions on 1 GPU
bash ./tools/inference.sh configs/video_knet_kitti_step/video_knet_s3_r50_rpn_1x_kitti_step_sigmoid_stride2_mask_embed_link_ffn_joint_train.py $CHECKPOINT $RESULTS_DIR
```

Colored images are also dupmed for debugging purposes.

The above command use the original Video K-Net architecture. To generate predictions for a different architecture, change the configuration file.

2. Evaluate predictions according to STQ and VPQ metrics.

```bash
# evaluate STQ
bash ./tools/evaluate_stq.sh $RESULTS_DIR 
```

```bash
# evaluate VPQ
bash ./tools/evaluate_vpq.sh $RESULTS_DIR
```

## Results and Visualizations


### Ablation Study: Video K-Net Architecture (Trained From Scratch)

| Adaptive kernel update strategy  | STQ      | AQ       | SQ       | VPQ      |  
|----------------------------------|----------|----------|----------|----------|
| Original                         | 53.9     | **50.0** | 58.2     | 38.6     |
| Concatenation                    | 51.0     | 48.1     | 54.1     | 37.2     |
| Skip connections                 | **54.3** | 49.7     | **59.4** | **38.9** |
| Concatenation + skip connections | 52.0     | 48.5     | 55.8     | 38.7     |
| MLP                              | 52.2     | 48.4     | 56.3     | 38.1     |


### Ablation Study: Video K-Net Architecture (Pretrained on Cityscapes-STEP)


### Ablation Study: Temporal Neighborhood for Sampling Reference Images

| Temporal Neighborhood | STQ      | AQ       | SQ       | VPQ      |  
|-----------------------|----------|----------|----------|----------|
| Original: [-2, 2]     | 67.5     | 68.7     | 66.3     | **48.3** |
| Causal: [-2, 0]       | 67.5     | **70.3** | 64.8     | 47.4     |
| More local: [-1, 1]   | 66.9     | 67.7     | 66.2     | 47.6     |
| More global: [-3, 3]  | **68.1** | 68.9     | **67.3** | 48.1     |


### Visualization
**TODO**


## Acknowledgement

Our code is based on the PyTorch implementation of [Video K-Net](https://github.com/lxtGH/Video-K-Net).

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{li2022videoknet,
  title={Video k-net: A simple, strong, and unified baseline for video segmentation},
  author={Li, Xiangtai and Zhang, Wenwei and Pang, Jiangmiao and Chen, Kai and Cheng, Guangliang and Tong, Yunhai and Loy, Chen Change},
  booktitle={CVPR},
  year={2022}
}

@article{zhang2021k,
  title={K-net: Towards unified image segmentation},
  author={Zhang, Wenwei and Pang, Jiangmiao and Chen, Kai and Loy, Chen Change},
  journal={NeurIPS},
  year={2021}
}
```