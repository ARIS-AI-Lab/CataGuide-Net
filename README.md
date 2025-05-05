# 🩺 CataGuide-Net: AI-Guided System for Cataract Surgery (https://aris-ai-lab.github.io/CataGuide-Net/)
CataGuide-Net is an AI-assisted surgical guidance framework designed to enhance cataract surgery through real-time perception and decision support. It integrates:

🎯 Instrument Segmentation – accurately identifies surgical tools and anatomical structures

🛠 Tip Detection – pinpoints tool tips to monitor critical movements

🔄 Stage Recognition – classifies surgical phases using temporal video models

🤖 Trajectory Guidance – provides skill-based feedback and expert trajectory imitation

The system supports intelligent training, surgical evaluation, and potential intraoperative assistance, especially for resource-limited settings.
### Cataract Instrument Segmentation and Tool tip detection

```bash
cd cataract_Seg
```

### Cataract Surgery Stage Recognition
```bash
cd Video_Classification
```

## Getting Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Trajectory Generator
```bash
python GAIL_train.py
```

You can find all our pre-trained model [here](https://drive.google.com/drive/folders/1XWkPpOvfDpVvim4MM7nbUQyge_vu9xJL)

