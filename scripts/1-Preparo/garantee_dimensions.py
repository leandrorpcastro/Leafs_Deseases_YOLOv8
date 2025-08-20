import cv2
from pathlib import Path
import os

BASE_DIR = Path("../../datasets")

DATASET_NAMES = [
    "Botoes_de_cafe",
    "Flores_de_cafe",
    "Folhas_doentes"
]
TARGET_SIZE = (640, 640)
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def resize_images_in_place():
    """
    Percorre os datasets especificados, encontra as imagens e as redimensiona
    para o TARGET_SIZE, sobrescrevendo os arquivos originais.
    """
    if not BASE_DIR.is_dir():
        print(f"❌ Erro: O diretório base '{BASE_DIR}' não foi encontrado.")
        print("Certifique-se de que o script está na pasta correta.")
        return

    print("--- Iniciando o processo de redimensionamento de imagens ---")

    # Contadores para o relatório final
    total_processed = 0
    total_skipped = 0
    total_errors = 0

    # Itera sobre cada nome de dataset fornecido
    for name in DATASET_NAMES:
        dataset_path = BASE_DIR / name
        if not dataset_path.is_dir():
            print(f"⚠️ Aviso: Dataset '{name}' não encontrado em '{BASE_DIR}'. Pulando.")
            continue

        print(f"\n📁 Processando dataset: '{name}'")
        images_root_path = dataset_path / "images"

        if not images_root_path.is_dir():
            print(f"  -> ⚠️ Pasta 'images' não encontrada em '{dataset_path}'. Pulando.")
            continue

        # Procura por subpastas como 'train', 'valid', 'test' dentro de 'images'
        for split_dir in images_root_path.iterdir():
            if split_dir.is_dir():
                print(f"  -> Processando split: '{split_dir.name}'")

                # Encontra todos os arquivos de imagem recursivamente
                image_paths = [p for p in split_dir.rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS]

                if not image_paths:
                    print("    -> Nenhuma imagem encontrada.")
                    continue

                for img_path in image_paths:
                    try:
                        # Carrega a imagem
                        img = cv2.imread(str(img_path))
                        if img is None:
                            print(f"    -> ❌ Erro ao ler a imagem: {img_path.name}")
                            total_errors += 1
                            continue

                        # Verifica as dimensões atuais
                        height, width, _ = img.shape
                        if (width, height) == TARGET_SIZE:
                            # print(f"    -> ⏭️ Imagem '{img_path.name}' já está no tamanho correto. Pulando.")
                            total_skipped += 1
                            continue

                        # Redimensiona a imagem
                        resized_img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

                        # Sobrescreve o arquivo original
                        cv2.imwrite(str(img_path), resized_img)
                        print(f"    -> ✅ Imagem '{img_path.name}' redimensionada para {TARGET_SIZE}.")
                        total_processed += 1

                    except Exception as e:
                        print(f"    -> ❌ Erro inesperado ao processar '{img_path.name}': {e}")
                        total_errors += 1

    print("\n--- Processo Concluído! ---")
    print(f"Resumo:")
    print(f"  Imagens redimensionadas: {total_processed}")
    print(f"  Imagens já no tamanho correto (puladas): {total_skipped}")
    print(f"  Erros encontrados: {total_errors}")


if __name__ == "__main__":
    print("⚠️ Atenção: Este script sobrescreve os arquivos de imagem originais.")
    print("É altamente recomendável fazer um BACKUP da sua pasta 'datasets' antes de continuar.")
    confirm = input("Você fez um backup e deseja continuar? (s/n): ")

    if confirm.lower() == 's':
        resize_images_in_place()
    else:
        print("Operação cancelada pelo usuário.")