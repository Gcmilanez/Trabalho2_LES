import pandas as pd
from sklearn.datasets import fetch_covtype
import numpy as np

print("Baixando Forest Covertype (COMPLETO)...")
print("⚠️  Dataset REAL com 581k amostras!\n")

# Baixa o dataset completo
data = fetch_covtype()

print(f"✓ Download completo!")
print(f"  Shape: {data.data.shape}")

# Converte para DataFrame
df = pd.DataFrame(data.data)
df['target'] = data.target

print("\nSalvando covertype_full.csv...")
df.to_csv('covertype_full.csv', index=False, header=False)

print(f"\n✅ Dataset salvo: covertype_full.csv")
print(f"  Amostras: {len(df):,}")
print(f"  Features: {data.data.shape[1]}")
print(f"  Classes: {df['target'].nunique()}")
print(f"  Tamanho: ~{df.memory_usage(deep=True).sum() / 1e6:.0f}MB")
