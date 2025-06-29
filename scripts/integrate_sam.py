from ultralytics import YOLO
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt

# Caminhos
image_path = 'teste.jpg'
yolo_model_path = '../runs/leaf-disease-yolo-treino1/weights/best.pt'
sam_checkpoint = '../models/sam_vit_h_4b8939.pth'

# 1. Carrega imagem
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 2. Carrega YOLOv8
model = YOLO(yolo_model_path)
results = model(image_rgb)

# 3. Pega as caixas (bounding boxes)
boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

# 4. Carrega o SAM
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# 5. Segmenta cada bbox com SAM
for i, box in enumerate(boxes):
    input_box = np.array(box)
    masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)

    # 6. Mostra a máscara de maior score
    mask = masks[np.argmax(scores)]

    plt.figure(figsize=(5, 5))
    plt.imshow(image_rgb)
    plt.imshow(mask, alpha=0.5)  # Máscara semi-transparente
    plt.title(f"Segmentação SAM - Box {i}")
    plt.axis('off')
    plt.show()
