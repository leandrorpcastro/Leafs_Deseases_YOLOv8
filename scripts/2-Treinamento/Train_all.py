from ultralytics import YOLO
import torch
import os

# --- CONFIGURAÇÃO CENTRAL DE TREINAMENTO ---

CONFIGS = {
    'botoes': {
        'task_type': 'detect',
        'data_yaml': '../../datasets/Botoes_de_cafe/data.yaml',
        'base_model': 'yolov8n.pt',  # Modelo para DETECÇÃO
        'project_name': 'treino_botoes_cafe',
        'experiment_name': 'botoes_exp1'
    },
    'folhas': {
        'task_type': 'detect',
        'data_yaml': '../../datasets/Folhas_doentes/data.yaml',
        'base_model': 'yolov8n.pt',  # Modelo para DETECÇÃO
        'project_name': 'treino_folhas_doentes',
        'experiment_name': 'folhas_exp1'
    },
    'flores': {
        'task_type': 'segment',
        'data_yaml': '../../datasets/Flores_de_cafe/data.yaml',
        'base_model': 'yolov8n-seg.pt',  # Modelo para SEGMENTAÇÃO!
        'project_name': 'treino_flores_cafe',
        'experiment_name': 'flores_exp1'
    }
}

# --- PARÂMETROS GERAIS DE TREINO ---
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 8


def treinar_modelos(modelos_a_treinar: list):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Usando dispositivo: {device.upper()}")

    for nome_modelo in modelos_a_treinar:
        if nome_modelo not in CONFIGS:
            print(f"⚠️ Aviso: Configuração para '{nome_modelo}' não encontrada. Pulando.")
            continue

        config = CONFIGS[nome_modelo]
        print("\n" + "=" * 50)
        print(f"🔥 Iniciando treinamento para: {nome_modelo.upper()} 🔥")
        print(f"   - Tipo de Tarefa: {config['task_type']}")
        print(f"   - Dataset: {config['data_yaml']}")
        print(f"   - Modelo Base: {config['base_model']}")
        print("=" * 50 + "\n")

        model = YOLO(config['base_model'])
        model.to(device)

        model.train(
            data=config['data_yaml'],
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            project=config['project_name'],  # Cria uma pasta separada para cada projeto
            name=config['experiment_name'],  # Nome do experimento dentro da pasta do projeto
            pretrained=True
        )

        print(f"\n✅ Treinamento para '{nome_modelo.upper()}' concluído!")


# --- EXECUÇÃO PRINCIPAL ---
if __name__ == '__main__':

    #Escolha aqui quais modelos você quer treinar!
    modelos_para_treinar = ['botoes', 'folhas', 'flores']

    treinar_modelos(modelos_para_treinar)