# Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward.
Implement with python=3.x https://github.com/KaiyangZhou/pytorch-vsumm-reinforce

<div align="center">
  <img src="img/pipeline.jpg" alt="train" width="80%">
</div>

## Requirement
python=3.x

Pytorch

GPU

tabulate

## Get started
```bash
git clone https://github.com/TorRient/Video-Summarization-Pytorch
cd Video-Summarization-Pytorch
mkdir dataset
```

1. Prepare dataset

Put your videos in folder dataset
```bash
python create_data.py --input dataset --output dataset/data.h5
```

2. Make splits
```bash
python create_split.py -d dataset/data.h5 --save-dir dataset --save-name splits  --num-splits 5
```

## How to train
```bash
python train_video_summarisation.py -d dataset/data.h5
```

## How to test
```bash
python video_summary.py --input path-to-video
```
Video summarization will be saved folder output/
