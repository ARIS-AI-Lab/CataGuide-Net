# Instrument Segmentation and Tip Detection
- Instrument Segmentation involves identifying and classifying each pixel of surgical tools within an image or video frame. This allows the system to understand the precise shape, location, and type of instruments being used.

- Tip Detection focuses on accurately locating the working end (tip) of a surgical instrument, which is critical for tracking movements and guiding surgical procedures.

## 🚀 Getting Started
### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset (Cataract-1k)
Download and organize the Download and organize the [Cataract-1k](https://www.synapse.org/Synapse:syn53404507) dataset in the following structure:
```angular2html
data/
├── images/                 # RGB frames
├── masks/                  # Ground truth masks (same filenames as images)
└── keypoints.json          # Tool tip annotations (optional)
```

### 3. Train Segmentation Model and Tip detection branch
```
python train.py
```

### 📚 Citation
```bibtex
@article{GAN2021118,
title = {Light-weight network for real-time adaptive stereo depth estimation},
journal = {Neurocomputing},
volume = {441},
pages = {118-127},
year = {2021},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2021.02.014},
url = {https://www.sciencedirect.com/science/article/pii/S0925231221002599},
author = {Wanshui Gan and Pak Kin Wong and Guokuan Yu and Rongchen Zhao and Chi Man Vong},
keywords = {Depth estimation, Domain adaptation, Neural network, Stereo matching, Self-supervised learning},
abstract = {Self-supervised learning methods have been proved effective in the task of real-time stereo depth estimation with the requirement of lower memory space and less computational cost. In this paper, a light-weight adaptive network (LWANet) is proposed by combining the self-supervised learning method to perform online adaptive stereo depth estimation for low computation cost and low GPU memory space. Instead of a regular 3D convolution, the pseudo 3D convolution is employed in the proposed light-weight network to aggregate the cost volume for achieving a better balance between the accuracy and the computational cost. Moreover, based on U-Net architecture, the downsample feature extractor is combined with a refined convolutional spatial propagation network (CSPN) to further refine the estimation accuracy with little memory space and computational cost. Extensive experiments demonstrate that the proposed LWANet effectively alleviates the domain shift problem by online updating the neural network, which is suitable for embedded devices such as NVIDIA Jetson TX2. The relevant codes are available at https://github.com/GANWANSHUI/LWANet}
}
```