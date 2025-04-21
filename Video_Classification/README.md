# 🎬 Cataract Surgery stage Recognition

Cataract Surgery Stage Recognition aims to automatically classify each frame or segment of a surgical video into its corresponding procedural stage. Accurate stage recognition is crucial for real-time decision support, surgical training, and workflow optimization.

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Prepare Dataset ([Cataract](https://ftp.itec.aau.at/datasets/ovid/cat-101/))
This project uses the **Cataract-101** dataset. Preprocessing involves two steps:
```bash
python preprocessing/train_test_split.py
```
```bash
python preprocessing/create_npy.py
```

### 3. Train the Model
```bash
python train.py
```

### 4. Evaluation
```bash
python eval/visionlization.py
```

### 📚 Citation
```bibtex
@misc{bertasius2021spacetimeattentionneedvideo,
      title={Is Space-Time Attention All You Need for Video Understanding?}, 
      author={Gedas Bertasius and Heng Wang and Lorenzo Torresani},
      year={2021},
      eprint={2102.05095},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2102.05095}, 
}
```