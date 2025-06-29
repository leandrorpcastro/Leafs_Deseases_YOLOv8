from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import torch

# --- CONFIGURAÇÕES ---
# Caminhos
image_path = '../datasets/val/images/f7a71ecf-bicho_mineiro98.jpg'
yolo_model_path = '../runs/leaf-disease-yolo-treino1/weights/best.pt'
sam_checkpoint_path = '../models/sam_vit_h_4b8939.pth'

# Garante que saída vá para uma pasta organizada com timestamp
timestamp = datetime.now().strftime("%d-%m-%H-%M-%S")
output_dir = Path("../resultados_SAM")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"resultado-{timestamp}.png"

# --- CARREGAR IMAGEM ---
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# --- CARREGAR MODELO YOLO ---
yolo_model = YOLO(yolo_model_path)
yolo_results = yolo_model(image_path, imgsz=640, conf=0.25, iou=0.5)[0]
boxes = yolo_results.boxes.xyxy.cpu().numpy().astype(int)
classes = yolo_results.boxes.cls.cpu().numpy().astype(int)
confidences = yolo_results.boxes.conf.cpu().numpy()

# --- RESULTADOS DA DETECÇÂO ---
print("Classes detectadas:", np.unique(classes))
# Verifica se só detectou a classe 0 (Folha)
if np.array_equal(np.unique(classes), [0]) or np.array_equal(np.unique(classes), []):
    print("⚠️ Apenas folhas detectadas. Nenhuma doença foi identificada.")
    exit()

# --- CARREGAR SAM ---
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
sam.to(device)
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# --- VARIÁVEIS PARA CÁLCULO ---
mask_folha = None
mask_doenca_total = np.zeros(image_rgb.shape[:2], dtype=bool)

# --- PROCESSAR CADA BBOX ---
for class_id, box in zip(classes, boxes):
    x1, y1, x2, y2 = box

    if class_id == 0:  # Folha
        input_point = np.array([[(x1 + x2) // 2, (y1 + y2) // 2]])
        input_label = np.array([1])  # foreground

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        best_mask = masks[np.argmax(scores)]
        mask_folha = best_mask

    else:  # Doença
        # Aqui só faz a predição com a bbox
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        best_mask = masks[np.argmax(scores)]
        mask_doenca_total = np.logical_or(mask_doenca_total, best_mask)


# --- CÁLCULO FINAL ---
if mask_folha is not None:
    area_folha = np.sum(mask_folha)
    area_doenca = np.sum(np.logical_and(mask_doenca_total, mask_folha))  # doença dentro da folha
    percentual_afetado = (area_doenca / area_folha) * 100
else:
    percentual_afetado = 0.0

# --- VISUALIZAÇÃO ---
plt.figure(figsize=(8, 8))
plt.imshow(image_rgb)
if mask_folha is not None:
    plt.imshow(mask_folha, alpha=0.5, cmap='YlGn')
plt.imshow(mask_doenca_total, alpha=0.7, cmap='Reds')
plt.title(f"Área afetada: {percentual_afetado:.2f}%")
plt.axis('off')
plt.tight_layout()
plt.savefig(output_file)

print(f"✅ Resultado salvo em: {output_file.resolve()}")
