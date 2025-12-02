"""
Script para gerar dataset Iris usando scikit-learn
Gera arquivo CSV que será usado pelo programa C++
"""

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

def generate_iris_csv():
    # Carregar dataset Iris do scikit-learn
    print("Carregando dataset Iris do scikit-learn...")
    iris = load_iris()
    
    # Criar DataFrame
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    df['species'] = iris.target
    
    # Informações sobre o dataset
    print(f"\nDataset Iris:")
    print(f"  - Amostras: {len(df)}")
    print(f"  - Features: {len(iris.feature_names)}")
    print(f"  - Classes: {len(iris.target_names)}")
    print(f"\nNomes das features:")
    for i, name in enumerate(iris.feature_names):
        print(f"  {i}: {name}")
    print(f"\nNomes das classes:")
    for i, name in enumerate(iris.target_names):
        print(f"  {i}: {name}")
    
    # Salvar em CSV
    output_file = "iris_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDataset salvo em: {output_file}")
    
    # Mostrar primeiras linhas
    print(f"\nPrimeiras 5 linhas:")
    print(df.head())
    
    print(f"\nDistribuição das classes:")
    print(df['species'].value_counts().sort_index())

if __name__ == "__main__":
    generate_iris_csv()
