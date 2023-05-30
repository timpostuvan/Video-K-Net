# Ablation Studies of Video K-Net: A Simple, Strong, and Unified Baseline for Video Segmentation

We explore modifications of Video K-Net, a simple, strong, and unified framework for fully end-to-end dense video segmentation. Video K-Net is built upon K-Net, a method of unifying image segmentation via a group of learnable kernels.

This project contains the training and evaluation code of Video K-Net for Video Panoptic Segmentation (VPS) on KITTI-STEP dataset.


### Overview of Contributions

In our work, we try to improve Video K-Net architecture and training regime to boost performance on VPS task. More specifically, we focus on the setting without an abundance of data, such as in the case of training on KITTI-STEP dataset. Our contributions are twofold:

1. **We ablate different adaptive feature update strategies for K-Net**: concatenation, skip connections, concatenation + skip connections, and MLP. Concatenation could potentially improve performance as information from the assembled features and kernels is more explicitly presented to the model, in contrast, to elementwise multiplication in the original architecture. On the other hand, skip connections remove the need for learning two update functions, and enforce that the model learns only a single "difference" update, which is more sample efficient (fewer parameters). To examine the possibility that the current model is not expressive enough, we also try a variant where all linear layers are substituted with multilayer perceptrons (MLPs) with two layers.

2. **We ablate different temporal neighborhoods for sampling reference images in Video K-Net**: more local neighborhood, more global neighborhood, and causal neighborhood. A more local neighborhood could encourage the model to focus on semantic segmentation, while a more global neighborhood might make the model focus more on tracking consistency. Furthermore, a causal neighborhood could improve object tracking as it might better learn the dynamics of objects (the reference image is always in the past of the key image).


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

# 4. Install Cython and other dependencies
pip install Cython==0.29.34
pip install -r requirements.txt
```

For instructions on how to obtain and properly prepare the datasets, see the [DATASET.md](https://github.com/timpostuvan/Video-K-Net/blob/main/DATASET.md).


### Checkpoints of Pretrained K-Net Models and Trained Video K-Net Models

We provide checkpoints of pre-trained and trained models. The pre-trained K-Net models can be used as initialization to train the Video K-Net, while the trained Video K-Net models are ready for playing and testing.

Google Drive Link: [here](https://drive.google.com/drive/folders/1DQvusMenJ01Am53-6UYAWaKx7naFrwOi?usp=sharing)


### Training on KITTI-STEP

1. Pretrain K-Net on Cityscapes-STEP dataset. This step is very important to improve the segmentation performance.

Cityscape-STEP follows the format of STEP: 17 stuff classes and 2 thing classes. 

```bash
# train Cityscapes STEP panoptic segmentation model on 3 GPUs
bash ./tools/dist_train.sh configs/knet_cityscapes_step/knet_s3_r50_fpn.py 3 $WORK_DIR --no-validate
```

2. Train the Video K-Net on KITTI-STEP. We have provided checkpoints of the pre-trained K-Net models on Cityscapes-STEP.

```bash
# train Video K-Net on KITTI-step with 3 GPUs from pre-trained checkpoint
bash ./tools/dist_train.sh configs/video_knet_kitti_step/video_knet_s3_r50_rpn_1x_kitti_step_sigmoid_stride2_mask_embed_link_ffn_joint_train.py 3 $WORK_DIR --no-validate --load-from $CHECKPOINT
```

It is also possible to train Video K-Net from scratch, however, this results in significantly inferior performance.

```bash
# train Video K-Net on KITTI-step with 3 GPUs from scratch
bash ./tools/dist_train.sh configs/video_knet_kitti_step/video_knet_s3_r50_rpn_1x_kitti_step_sigmoid_stride2_mask_embed_link_ffn_joint_train.py 3 $WORK_DIR --no-validate
```

The above commands use the original Video K-Net architecture. To train a modified architecture or in a different experimental setting, change the configuration file.


### Evaluation on KITTI-STEP

1. Generate predictions on the validation set.

```bash
# generate predictions on 1 GPU
bash ./tools/inference.sh configs/video_knet_kitti_step/video_knet_s3_r50_rpn_1x_kitti_step_sigmoid_stride2_mask_embed_link_ffn_joint_train.py $CHECKPOINT $RESULTS_DIR
```

Colored images are also dumped for visualization purposes.

The above command uses the original Video K-Net architecture. To generate predictions for a different architecture, change the configuration file.

2. Evaluate predictions according to STQ and VPQ metrics.

```bash
# evaluate STQ
bash ./tools/evaluate_stq.sh $RESULTS_DIR 
```

```bash
# evaluate VPQ
bash ./tools/evaluate_vpq.sh $RESULTS_DIR
```


### Visualization

To visualize predictions, run the following command that generates colorized images (in `colorized` folder) and GIFs of colorized images (in `colorized_gif` folder).

```bash
# visualize predictions
bash ./tools/visualize.sh $RESULTS_DIR
```


## Results and Visualizations

**Common experimental setup:**  We use the distributed training framework with 3 GPUs. Each minibatch has one image per GPU. In all our experiments, we use ResNet as the backbone, while other layers are initialized by Xavier initialization. The optimizer is AdamW with a weight decay of 0.0001. Unless specified otherwise, we follow the exact setting from the original Video K-Net paper. We evaluate all our models according to two widely used metrics: Video  Panoptic Quality (VPQ) and Segmentation and Tracking Quality (STQ). VPQ mainly focuses on mask proposal levels like Panoptic Quality (PQ) with different window sizes and threshold parameters, while STQ emphasizes overall pixel-level segmentation and tracking. STQ is computed as the geometric mean of two metrics: Segmentation Quality (SQ) and Association Quality (AQ). SQ measures the pixel-level segmentation results in a video clip, while AQ measures the pixel-level tracking. We adopt STQ as the primary evaluation metric due to its explainability.


### Ablation Study: Video K-Net Architecture (Trained From Scratch)

**Experimental setup:** In this experiment, we substitute the original adaptive feature update strategy with concatenation, skip connections, concatenation + skip connections, and MLPs. In the interest of computational cost, the models are trained only on KITTI-STEP without pretraining on Cityscapes-STEP.

| Adaptive feature update strategy  | STQ      | AQ       | SQ       | VPQ      |  
|-----------------------------------|----------|----------|----------|----------|
| Original                          | 53.9     | **50.0** | 58.2     | 38.6     |
| Concatenation                     | 51.0     | 48.1     | 54.1     | 37.2     |
| Skip connections                  | **54.3** | 49.7     | **59.4** | **38.9** |
| Concatenation + skip connections  | 52.0     | 48.5     | 55.8     | 38.7     |
| MLP                               | 52.2     | 48.4     | 56.3     | 38.1     |

**Results:** The original architecture is a very strong baseline and only the architecture with skip connections outperform it. Concatenation only harms performance, no matter whether it is employed together with skip connections or not. Furthermore, the original's architecture has already enough capacity, and introducing 2-layer MLPs with more parameters performs inferiorly. The best architecture is the one with skip connections as it outperforms all others by more than 0.3% according to STQ and VPQ.


### Ablation Study: Video K-Net Architecture (Pre-trained on Cityscapes-STEP)

**Experimental setup:** In this experiment, we compare the original architecture with skip connections feature update (the best-performing architecture in the previous experiment) in a setting with additional pretraining on Cityscapes-STEP.

| Adaptive feature update strategy  | STQ      | AQ       | SQ       | VPQ      |  
|-----------------------------------|----------|----------|----------|----------|
| Original                          | 64.8     | 64.9     | **64.6** | **46.2** |
| Skip connections                  | **66.0** | **67.6** | 64.4     | 45.7     |

**Results:** Here, both architectures achieve similar results, however, the architecture with skip connections seems to be slightly better. Its STQ score is greater by 1.2%, while only 0.5% worse according to VPQ metric. It is especially better in object tracking as its AQ score is greater by almost 3%.


### Ablation Study: Temporal Neighborhood for Sampling Reference Images

**Experimental setup:** In this experiment, we compare the original temporal neighborhoods for sampling of reference images (interval [-2, 2]) with causal neighborhood (interval [-2, 0]), more local neighborhood (interval [-1, 1]), and more global neighborhood (interval [-3, 3]). All models are trained on KITTI-STEP and pre-trained on Cityscapes-STEP.


| Temporal Neighborhood | STQ      | AQ       | SQ       | VPQ      |  
|-----------------------|----------|----------|----------|----------|
| Original: [-2, 2]     | 67.5     | 68.7     | 66.3     | **48.3** |
| Causal: [-2, 0]       | 67.5     | **70.3** | 64.8     | 47.4     |
| More local: [-1, 1]   | 66.9     | 67.7     | 66.2     | 47.6     |
| More global: [-3, 3]  | **68.1** | 68.9     | **67.3** | 48.1     |

**Results:** Sampling reference images from a broader neighborhood seems to be important since more local neighborhood consistently performs worse than the original and more global neighborhoods. Causal neighborhood outperforms all other alternatives according to AQ, which indicates that the model can better learn object dynamics. When comparing the original neighborhood with the more global one, there is no significant difference. One performs better according to STQ, while the other comes on top according to VPQ.


### Visualization

#### Original Model:
Input RGB images (left), semantic segmentation masks (middle), instance segmentation masks (right)
![Alt Text](./figs/video_knet_step_pretrained.gif)

#### Our Best Model with Skip Connections:
Input RGB images (left), semantic segmentation masks (middle), instance segmentation masks (right)
![Alt Text](./figs/video_knet_step_pretrained_skip.gif)


## Conclusion

Overall, in the setting without plenty of data, skip connections outperform the original architecture (with and without pretraining). This can be also observed from the visualizations as the architecture with skip connections better tracks objects. On the other hand, the exact size of temporal neighborhood for sampling reference images is not crucial, as long as the neighborhood is broad enough.


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
