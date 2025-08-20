import json
import os
from pathlib import Path

# --- Configuração ---
# Caminho para o seu arquivo COCO JSON
json_file_path = '../valid/_annotations.coco.json'

# Diretório onde você quer salvar os labels do YOLO
output_dir = '../../datasets/Flores_de_cafe/valid/labels'

# Cria o diretório de saída se ele não existir
Path(output_dir).mkdir(parents=True, exist_ok=True)

# 1. Carregar o arquivo JSON
with open(json_file_path, 'r') as f:
    coco_data = json.load(f)

# 2. Criar mapeamentos para facilitar a busca
images = {img['id']: img for img in coco_data['images']}
# Mapeia category_id para um índice 0, 1, 2... que é o que o YOLO espera
categories = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}
category_names = {i: cat['name'] for i, cat in enumerate(coco_data['categories'])}

# 3. Processar cada anotação
annotations = coco_data['annotations']
total_annotations = len(annotations)
print(f"Encontradas {total_annotations} anotações para processar...")

# Dicionário para agrupar anotações por imagem
annotations_by_image = {}
for ann in annotations:
    img_id = ann['image_id']
    if img_id not in annotations_by_image:
        annotations_by_image[img_id] = []
    annotations_by_image[img_id].append(ann)

# 4. Gerar os arquivos de label
file_count = 0
for img_id, anns in annotations_by_image.items():
    if img_id not in images:
        print(f"Aviso: Imagem com ID {img_id} não encontrada na seção 'images'. Pulando.")
        continue

    img_info = images[img_id]
    img_width = img_info['width']
    img_height = img_info['height']

    # Define o nome do arquivo de label .txt
    img_filename = Path(img_info['file_name'])
    label_filename = img_filename.with_suffix('.txt').name
    label_filepath = Path(output_dir) / label_filename

    yolo_lines = []

    for ann in anns:
        # Pega o índice da classe (0, 1, 2...)
        class_index = categories[ann['category_id']]

        # Pega as coordenadas de segmentação
        # O formato COCO pode ter múltiplos polígonos, aqui pegamos o primeiro
        segmentation = ann['segmentation'][0]

        # Normaliza as coordenadas
        normalized_coords = []
        for i in range(0, len(segmentation), 2):
            x = segmentation[i]
            y = segmentation[i + 1]
            x_norm = x / img_width
            y_norm = y / img_height
            normalized_coords.append(f"{x_norm:.6f}")  # formata com 6 casas decimais
            normalized_coords.append(f"{y_norm:.6f}")

        # Monta a linha no formato YOLO
        yolo_line = f"{class_index} {' '.join(normalized_coords)}"
        yolo_lines.append(yolo_line)

    # Salva todas as anotações para esta imagem no arquivo .txt
    with open(label_filepath, 'w') as f:
        f.write('\n'.join(yolo_lines))
    file_count += 1

print("\n--- Conversão Concluída! ---")
print(f"Total de {file_count} arquivos de label gerados em '{output_dir}'.")
print("\nMapeamento de Classes:")
for idx, name in category_names.items():
    print(f"  Classe {idx}: {name}")