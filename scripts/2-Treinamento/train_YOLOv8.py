from ultralytics import YOLO

# 🚀 Carregar modelo pré-treinado (pode escolher yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
model = YOLO('../../models/yolov8n.pt')

# 🏗️ Caminho para o seu arquivo data.yaml
DATA_YAML_PATH = '../../arqs_data_yaml/data_disease.yaml'

# 🔧 Configurações de treino
model.train(
    data=DATA_YAML_PATH,   # dataset
    epochs=100,            # número de épocas
    imgsz=640,             # tamanho das imagens (640x640 padrão)
    batch=8,               # batch size — ajuste conforme sua GPU/CPU
    project='runs',        # pasta onde os resultados vão ficar
    name='leaf-disease-yolo',  # nome do experimento
    pretrained=True,       # usa pesos pré-treinados
)
