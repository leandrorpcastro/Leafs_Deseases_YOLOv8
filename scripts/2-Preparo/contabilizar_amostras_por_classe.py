import os
from collections import defaultdict

# Caminho onde estão os arquivos de label
caminho_labels = "datasets/val/labels"

# Contadores
ocorrencias_por_classe = defaultdict(int)
arquivos_por_classe = defaultdict(int)

# Percorre todos os arquivos .txt na pasta
for nome_arquivo in os.listdir(caminho_labels):
    if nome_arquivo.endswith(".txt"):
        caminho_arquivo = os.path.join(caminho_labels, nome_arquivo)

        classes_no_arquivo = set()
        with open(caminho_arquivo, 'r') as f:
            for linha in f:
                partes = linha.strip().split()
                if partes:
                    classe = int(partes[0])
                    ocorrencias_por_classe[classe] += 1
                    classes_no_arquivo.add(classe)

        # Registra que este arquivo contém essa(s) classe(s)
        for classe in classes_no_arquivo:
            arquivos_por_classe[classe] += 1

# Exibe os resultados
print("Resumo das classes encontradas:\n")
for classe in sorted(ocorrencias_por_classe.keys()):
    print(f"Classe {classe}:")
    print(f"  - Total de amostras: {ocorrencias_por_classe[classe]}")
    print(f"  - Aparece em {arquivos_por_classe[classe]} arquivos\n")
