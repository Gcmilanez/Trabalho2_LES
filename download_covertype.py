"""
Script para baixar e preparar o dataset Forest Cover Type
Dataset: 581.012 amostras, 54 features, 7 classes
"""

from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np

print("========================================================")
print("  Baixando Forest Cover Type Dataset")
print("  581.012 amostras, 54 features, 7 classes")
print("========================================================\n")

print("Baixando dataset (pode levar alguns minutos)...")
covtype = fetch_covtype(download_if_missing=True)

print(f"Dataset carregado: {covtype.data.shape[0]} amostras")
print(f"Features: {covtype.data.shape[1]}")
print(f"Classes: {np.unique(covtype.target)}\n")

# Converter para DataFrame
print("Preparando arquivo CSV...")
df = pd.DataFrame(covtype.data)

# Ajustar labels para começar de 0 (ao invés de 1-7)
labels = covtype.target - 1

# Adicionar coluna de labels
df['class'] = labels

# Salvar CSV
output_file = "covertype_dataset.csv"
print(f"Salvando em: {output_file}")
df.to_csv(output_file, index=False)

print(f"\n✅ Dataset salvo com sucesso!")
print(f"   Arquivo: {output_file}")
print(f"   Tamanho: {df.shape[0]} amostras x {df.shape[1]} colunas")
print(f"   Classes: 0-6 (7 tipos de cobertura florestal)")
print("========================================================")
