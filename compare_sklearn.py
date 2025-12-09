import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import subprocess
import os

# Datasets para testar
datasets = {
    'optdigits': 'optdigits.csv',
    'adult': 'adult_dataset.csv',
    'skin': 'skin_segmentation.csv',
    'arrhythmia': 'arrhythmia.csv'
}

# Parâmetros iguais ao C++
MAX_DEPTH = 8
MIN_SAMPLES_SPLIT = 5
N_TREES = 50  # Para comparar com Random Forest depois

print("="*80)
print("COMPARAÇÃO: sklearn.tree.DecisionTreeClassifier vs C++ Otimizado")
print("="*80)

results = []

for dataset_name, dataset_file in datasets.items():
    if not os.path.exists(dataset_file):
        print(f"\n⚠️  Pulando {dataset_name} (arquivo não encontrado)")
        continue
    
    print(f"\n{'='*80}")
    print(f"📊 DATASET: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Carregar dados
    df = pd.read_csv(dataset_file)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    n_samples = len(X)
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    
    print(f"Amostras: {n_samples:,} | Features: {n_features} | Classes: {n_classes}")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # ========================================
    # SKLEARN DecisionTreeClassifier
    # ========================================
    print(f"\n🔵 sklearn.tree.DecisionTreeClassifier:")
    
    clf = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        random_state=42
    )
    
    # Treino
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Predição
    start = time.time()
    y_pred = clf.predict(X_test)
    pred_time = time.time() - start
    
    acc = accuracy_score(y_test, y_pred)
    
    print(f"   Treino:   {train_time:.3f}s")
    print(f"   Predição: {pred_time*1000:.2f}ms")
    print(f"   Acurácia: {acc*100:.2f}%")
    
    # ========================================
    # C++ OTIMIZADO (executar binário)
    # ========================================
    print(f"\n🟢 C++ Otimizado (com chunks):")
    
    # Treinar modelo C++
    cmd_train = [
        'forest_optimized_train.exe',
        dataset_file,
        str(n_samples),
        '1',  # 1 run
        f'models/temp_{dataset_name}.model'
    ]
    
    try:
        result = subprocess.run(cmd_train, capture_output=True, text=True, timeout=300)
        
        # Extrair tempo de treino do output
        cpp_train_time = None
        for line in result.stdout.split('\n'):
            if 'Tempo médio de treinamento' in line or 'Training time' in line:
                # Buscar número antes de 's' ou 'ms'
                import re
                match = re.search(r'([\d.]+)\s*s', line)
                if match:
                    cpp_train_time = float(match.group(1))
        
        # Predição
        cmd_pred = [
            'forest_optimized_predict.exe',
            f'models/temp_{dataset_name}.model',
            dataset_file,
            str(n_samples)
        ]
        
        result_pred = subprocess.run(cmd_pred, capture_output=True, text=True, timeout=60)
        
        cpp_pred_time = None
        cpp_acc = None
        for line in result_pred.stdout.split('\n'):
            if 'Tempo de predição' in line or 'Prediction time' in line:
                match = re.search(r'([\d.]+)\s*ms', line)
                if match:
                    cpp_pred_time = float(match.group(1))
            if 'Acurácia' in line or 'Accuracy' in line:
                match = re.search(r'([\d.]+)%', line)
                if match:
                    cpp_acc = float(match.group(1))
        
        if cpp_train_time:
            print(f"   Treino:   {cpp_train_time:.3f}s")
        if cpp_pred_time:
            print(f"   Predição: {cpp_pred_time:.2f}ms")
        if cpp_acc:
            print(f"   Acurácia: {cpp_acc:.2f}%")
        
        # Comparação
        if cpp_train_time and train_time:
            speedup = train_time / cpp_train_time
            improvement = ((train_time - cpp_train_time) / train_time) * 100
            print(f"\n📈 SPEEDUP: {speedup:.2f}× ({improvement:+.1f}%)")
        
    except Exception as e:
        print(f"   ❌ Erro ao executar C++: {e}")
        cpp_train_time = None
        cpp_pred_time = None
        cpp_acc = None
    
    # Salvar resultados
    results.append({
        'dataset': dataset_name,
        'samples': n_samples,
        'features': n_features,
        'classes': n_classes,
        'sklearn_train': train_time,
        'sklearn_pred': pred_time * 1000,
        'sklearn_acc': acc * 100,
        'cpp_train': cpp_train_time,
        'cpp_pred': cpp_pred_time,
        'cpp_acc': cpp_acc
    })

# Resumo final
print(f"\n{'='*80}")
print("📊 RESUMO COMPARATIVO")
print(f"{'='*80}\n")

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# Salvar CSV
df_results.to_csv('results/sklearn_vs_cpp.csv', index=False)
print(f"\n✅ Resultados salvos em: results/sklearn_vs_cpp.csv")
