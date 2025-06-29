import random
from pathlib import Path
import cv2
import shutil

# --- CONFIGURAÇÕES ---
dataset_root = Path("../datasets")
input_dir = dataset_root / "complete"
output_dir = dataset_root  # Onde serão criados os diretórios 'train/' e 'val/'

train_ratio = 0.8  # 80% treino, 20% validação
img_size = (640, 640)

# --- GARANTE QUE AS PASTAS DE SAÍDA EXISTEM ---
for split in ['train', 'val']:
    for sub in ['images', 'labels']:
        (output_dir / split / sub).mkdir(parents=True, exist_ok=True)

# --- LISTA TODAS AS IMAGENS ---
image_paths = list((input_dir / "images").glob("*.*"))
image_paths = [p for p in image_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
random.shuffle(image_paths)

# --- DIVIDE EM TREINO E VALIDAÇÃO ---
split_index = int(len(image_paths) * train_ratio)
train_images = image_paths[:split_index]
val_images = image_paths[split_index:]

# --- FUNÇÃO PARA COPIAR E REDIMENSIONAR IMAGENS E LABELS ---
def process_split(images, split):
    for img_path in images:
        # Redimensiona imagem
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Erro ao ler imagem: {img_path}")
            continue

        resized = cv2.resize(img, img_size)
        out_img_path = output_dir / split / "images" / img_path.name
        cv2.imwrite(str(out_img_path), resized)

        # Copia o label correspondente (se existir)
        label_path = input_dir / "labels" / (img_path.stem + ".txt")
        if label_path.exists():
            out_label_path = output_dir / split / "labels" / label_path.name
            shutil.copy(label_path, out_label_path)
        else:
            print(f"⚠️ Label não encontrado para {img_path.name}")

# --- EXECUTA PARA CADA CONJUNTO ---
process_split(train_images, 'train')
process_split(val_images, 'val')

print("✅ Dataset preparado com sucesso.")
