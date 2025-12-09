import pandas as pd
from sklearn.datasets import make_classification
import numpy as np

print("Gerando dataset MUITO GRANDE...")
print("⚠️  Isso vai criar ~4GB de dados CSV!\n")

# Dataset massivo para testar limites
X, y = make_classification(
    n_samples=1_000_000,      # 1 MILHÃO de amostras
    n_features=100,            # 100 features
    n_informative=80,
    n_redundant=10,
    n_classes=10,
    random_state=42,
    flip_y=0.01
)

print(f"✓ Dados gerados: {X.shape[0]:,} amostras × {X.shape[1]} features")
print(f"  Tamanho em memória: ~{X.nbytes / 1e9:.2f}GB")
print("\nSalvando massive_dataset.csv...")

# Salvar em chunks para não estourar memória
chunk_size = 100_000
total_chunks = len(X) // chunk_size

for i in range(0, len(X), chunk_size):
    end = min(i + chunk_size, len(X))
    df_chunk = pd.DataFrame(X[i:end])
    df_chunk['target'] = y[i:end]
    
    mode = 'w' if i == 0 else 'a'
    header = i == 0
    
    df_chunk.to_csv('massive_dataset.csv', mode=mode, header=header, index=False)
    
    progress = (end / len(X)) * 100
    print(f"  Progresso: {progress:.1f}% ({end:,}/{len(X):,})")

print(f"\n✅ Dataset salvo: massive_dataset.csv")
print(f"  Amostras: {len(X):,}")
print(f"  Features: {X.shape[1]}")
print(f"  Classes: {len(np.unique(y))}")
print(f"  Tamanho arquivo: ~4GB")
print("\n🔥 Use este dataset para stress test!")
