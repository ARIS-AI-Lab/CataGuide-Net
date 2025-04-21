# Instrument Segmentation and Tip Detection

## DataSet
In this research 

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset (Cataract-1k)
Download and organize the Download and organize the [Cataract-1k](#https://www.synapse.org/Synapse:syn53404507) dataset in the following structure:
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
