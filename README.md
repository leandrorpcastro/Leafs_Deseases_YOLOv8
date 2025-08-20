# Leaf Disease Detection and Analysis System with YOLOv8 and SAM Integration

This project implements an advanced computer vision system for detecting and analyzing leaf diseases in plants. It combines YOLOv8 object detection with the Segment Anything Model (SAM) to provide accurate disease detection, segmentation, and damage area calculation.

The system processes leaf images to identify multiple disease types including Miner, Rust, Cercospora, and Phoma. It not only detects the presence of diseases but also calculates the affected area percentage, providing quantitative analysis for plant health assessment. The integration of SAM enables precise segmentation of both leaves and diseased areas, making it a powerful tool for agricultural monitoring and disease management.

## Repository Structure
```
.
├── data.yaml                 # Dataset configuration file defining classes and paths
├── requirements.txt          # Project dependencies and versions
└── scripts/
    ├── calcula_dano.py      # Main script for damage area calculation using YOLO and SAM
    ├── contabilizar_amostras_por_classe.py  # Dataset analysis script for class distribution
    ├── infer_yolo.py        # YOLO inference script for disease detection
    ├── integrate_sam.py     # Integration script for YOLO and SAM models
    ├── prepare_dataset.py   # Dataset preparation and preprocessing script
    └── train_YOLOv8.py     # Training script for YOLOv8 model
```

## Usage Instructions
### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for optimal performance)
- Required Python packages:
  - opencv-python ~=4.11.0.86
  - matplotlib ~=3.10.3
  - numpy ~=2.3.0
  - ultralytics ~=8.3.156
  - torch ~=2.7.1
  - segment_anything ~=1.0

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Create and activate virtual environment
## For Linux/MacOS
python3 -m venv venv
source venv/bin/activate

## For Windows
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required model weights
## YOLOv8 weights will be downloaded automatically
## Download SAM weights manually and place in models/ directory
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P models/
```

### Quick Start

1. Prepare your dataset:
```bash
python scripts/prepare_dataset.py
```

2. Train the YOLOv8 model:
```bash
python scripts/train_YOLOv8.py
```

3. Run inference on an image:
```bash
python scripts/infer_yolo.py
```

### More Detailed Examples

1. Calculate disease damage area:

```python
from scripts.Inferencias.calcula_dano import calculate_damage

image_path = 'path/to/your/image.jpg'
result = calculate_damage(image_path)
print(f"Affected area percentage: {result['percentage']}%")
```

2. Analyze dataset distribution:
```bash
python scripts/contabilizar_amostras_por_classe.py
```

### Troubleshooting

Common issues and solutions:

1. CUDA Out of Memory
- Symptom: `RuntimeError: CUDA out of memory`
- Solution: Reduce batch size in train_YOLOv8.py
```python
model.train(
    batch=4,  # Reduce from 8 to 4 or lower
    ...
)
```

2. SAM Model Loading Issues
- Symptom: `FileNotFoundError: SAM checkpoint not found`
- Solution: Ensure SAM model weights are downloaded and placed in the correct directory
```bash
mkdir -p models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P models/
```

## Data Flow
The system processes images through a pipeline of detection and segmentation to produce detailed disease analysis.

```ascii
Input Image → YOLOv8 Detection → SAM Segmentation → Damage Calculation → Analysis Output
     ↓              ↓                    ↓                    ↓               ↓
[Raw Image] → [Bounding Boxes] → [Precise Masks] → [Area Calculations] → [Results]
```

Key component interactions:
1. YOLOv8 performs initial detection of leaves and diseases
2. Detection results are passed to SAM for precise segmentation
3. Segmentation masks are used to calculate affected areas
4. Results are combined for final analysis and visualization
5. All components share the same image preprocessing pipeline
6. Error handling is implemented at each stage
7. Results are cached for performance optimization