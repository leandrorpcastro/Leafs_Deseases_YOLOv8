from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURAÇÕES ---
image_path = '../datasets/val/images/f7a71ecf-bicho_mineiro98.jpg'
model_path = '../runs/leaf-disease-yolo-treino1/weights/best.pt'

# --- CARREGAR IMAGEM ---
image_bgr = cv2.imread(image_path)
if image_bgr is None:
    raise FileNotFoundError(f"Imagem não encontrada em: {image_path}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# --- CARREGAR MODELO ---
model = YOLO(model_path)

# --- INFERÊNCIA ---
#results = model.predict(source=image_rgb, imgsz=640, conf=0.05, iou=0.1, agnostic_nms=True)[0]
results = model.predict(source=image_path, imgsz=640, conf=0.25, iou=0.5, classes=[0,1,2,3,4], agnostic_nms=True)[0]
boxes = results.boxes.xyxy.cpu().numpy().astype(int)
classes = results.boxes.cls.cpu().numpy().astype(int)
confidences = results.boxes.conf.cpu().numpy()

# --- NOMES DAS CLASSES (ajuste conforme seu data.yaml) ---
class_names = ['Folhas', 'Miner', 'Rust', 'Cercospora', 'Phoma']
colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255)]

# --- DESENHAR DETECÇÕES ---
output = image_rgb.copy()
for cls, box, conf in zip(classes, boxes, confidences):
    label = f"{class_names[cls]} {conf:.2f}"
    color = colors[cls % len(colors)]
    x1, y1, x2, y2 = box
    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
    cv2.putText(output, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# --- MOSTRAR RESULTADO ---
plt.figure(figsize=(10, 8))
plt.imshow(output)
plt.title("Resultado da Inferência com YOLOv8")
plt.axis('off')
plt.tight_layout()
plt.show()

# --- DEBUG: Classes detectadas
print("Classes detectadas:", np.unique(classes))
