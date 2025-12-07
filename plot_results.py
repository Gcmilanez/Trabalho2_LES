import re
import matplotlib.pyplot as plt
import numpy as np

# Ler o arquivo Resultados.md
with open('Resultados.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Estruturas para armazenar dados
datasets = ['optdigits', 'adult', 'skin']
data = {
    'train': {ds: {'baseline': [], 'optimized': [], 'samples': []} for ds in datasets},
    'predict': {ds: {'baseline': [], 'optimized': [], 'samples': [], 'acc_base': [], 'acc_opt': []} for ds in datasets}
}

# Mapear tamanhos reais de samples
sample_mapping = {
    'optdigits': [500, 1000, 1797],
    'adult': [1000, 10000, 45222],
    'skin': [1000, 10000, 245057]
}

# Parse treinamento
train_pattern = r'(BASELINE|OTIMIZADO)\s+(\w+)\s+[\d,\.]+k:\s+([\d.]+)s'
for match in re.finditer(train_pattern, content):
    tipo, dataset, tempo = match.groups()
    tempo = float(tempo)
    
    if dataset in datasets:
        if tipo == 'BASELINE':
            data['train'][dataset]['baseline'].append(tempo)
            if len(data['train'][dataset]['samples']) < len(sample_mapping[dataset]):
                data['train'][dataset]['samples'].append(sample_mapping[dataset][len(data['train'][dataset]['baseline'])-1])
        else:
            data['train'][dataset]['optimized'].append(tempo)

# Parse predição
pred_pattern = r'(BASELINE|OTIMIZADO)\s+(\w+)\s+[\d,\.]+k:\s+([\d.]+)ms\s+\(([\d.]+)%\s+acc\)'
for match in re.finditer(pred_pattern, content):
    tipo, dataset, tempo, acc = match.groups()
    tempo = float(tempo)
    acc = float(acc)
    
    if dataset in datasets:
        if tipo == 'BASELINE':
            data['predict'][dataset]['baseline'].append(tempo)
            data['predict'][dataset]['acc_base'].append(acc)
            if len(data['predict'][dataset]['samples']) < len(sample_mapping[dataset]):
                data['predict'][dataset]['samples'].append(sample_mapping[dataset][len(data['predict'][dataset]['baseline'])-1])
        else:
            data['predict'][dataset]['optimized'].append(tempo)
            data['predict'][dataset]['acc_opt'].append(acc)

# Criar gráficos com barras verticais - SEPARADOS
colors = {'baseline': '#FF0000', 'optimized': '#00AA00'}  # Vermelho puro e Verde
dataset_names = {'optdigits': 'OptDigits', 'adult': 'Adult', 'skin': 'Skin Segmentation'}

# Linha 1: Treinamento
for idx, dataset in enumerate(datasets):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    samples = data['train'][dataset]['samples']
    baseline = data['train'][dataset]['baseline']
    optimized = data['train'][dataset]['optimized']
    
    x = np.arange(len(samples))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline, width, label='Baseline', color=colors['baseline'], alpha=0.8)
    bars2 = ax.bar(x + width/2, optimized, width, label='Otimizado', color=colors['optimized'], alpha=0.8)
    
    # Calcular speedup
    speedups = [b/o for b, o in zip(baseline, optimized)]
    
    ax.set_yscale('log')
    ax.set_xlabel('Número de Amostras', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tempo de Treinamento (s) - MENOR É MELHOR ⬇', fontsize=12, fontweight='bold', color='darkgreen')
    
    # Título com informações do dataset
    features_info = '64 features' if dataset == 'optdigits' else '14 features' if dataset == 'adult' else '3 features'
    ax.set_title(f'Treinamento - {dataset_names[dataset]} ({features_info})\nRandom Forest com Chunks de 100', 
                 fontsize=13, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s//1000}k' if s >= 1000 else str(s) for s in samples])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left', fontsize=11)
    
    # Adicionar speedup em cada par de barras
    for i, (b, o, sp) in enumerate(zip(baseline, optimized, speedups)):
        improvement = ((b - o) / b) * 100
        ax.text(i, max(b, o) * 1.15, f'{sp:.2f}×\n({improvement:.0f}%)', 
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'imgs/train_{dataset}.png', dpi=300, bbox_inches='tight')
    print(f'✅ Gráfico salvo: imgs/train_{dataset}.png')
    plt.close()

# Linha 2: Predição
for idx, dataset in enumerate(datasets):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    samples = data['predict'][dataset]['samples']
    baseline = data['predict'][dataset]['baseline']
    optimized = data['predict'][dataset]['optimized']
    
    x = np.arange(len(samples))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline, width, label='Baseline', color=colors['baseline'], alpha=0.8)
    bars2 = ax.bar(x + width/2, optimized, width, label='Otimizado', color=colors['optimized'], alpha=0.8)
    
    # Calcular speedup
    speedups = [b/o for b, o in zip(baseline, optimized)]
    
    ax.set_yscale('log')
    ax.set_xlabel('Número de Amostras', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tempo de Predição (ms) - MENOR É MELHOR ⬇', fontsize=12, fontweight='bold', color='darkgreen')
    
    # Título com informações do dataset
    features_info = '64 features' if dataset == 'optdigits' else '14 features' if dataset == 'adult' else '3 features'
    ax.set_title(f'Predição - {dataset_names[dataset]} ({features_info})\nRandom Forest com Chunks de 100', 
                 fontsize=13, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s//1000}k' if s >= 1000 else str(s) for s in samples])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left', fontsize=11)
    
    # Adicionar speedup em cada par de barras
    for i, (b, o, sp) in enumerate(zip(baseline, optimized, speedups)):
        improvement = ((b - o) / b) * 100
        ax.text(i, max(b, o) * 1.15, f'{sp:.2f}×\n({improvement:.0f}%)', 
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'imgs/predict_{dataset}.png', dpi=300, bbox_inches='tight')
    print(f'✅ Gráfico salvo: imgs/predict_{dataset}.png')
    plt.close()

print('\n📊 Total: 6 gráficos gerados (3 treino + 3 predição)')
print('='*80)

# Debug: Mostrar dados parseados
print('\n🔍 Dados de Predição Parseados:')
for dataset in datasets:
    print(f'\n{dataset.upper()}:')
    print(f'  Samples: {data["predict"][dataset]["samples"]}')
    print(f'  Baseline: {data["predict"][dataset]["baseline"]}')
    print(f'  Optimized: {data["predict"][dataset]["optimized"]}')

# Criar tabela de speedup
print('\n📊 RESUMO DE SPEEDUP:')
print('='*80)
for dataset in datasets:
    print(f'\n{dataset_names[dataset].upper()}:')
    print('-'*80)
    
    # Treinamento
    samples = data['train'][dataset]['samples']
    baseline_train = data['train'][dataset]['baseline']
    optimized_train = data['train'][dataset]['optimized']
    
    print('  TREINAMENTO:')
    for i, s in enumerate(samples):
        if i < len(baseline_train) and i < len(optimized_train):
            speedup = baseline_train[i] / optimized_train[i]
            improvement = ((baseline_train[i] - optimized_train[i]) / baseline_train[i]) * 100
            print(f'    {s:>7} samples: {baseline_train[i]:>8.2f}s → {optimized_train[i]:>8.2f}s | '
                  f'Speedup: {speedup:.3f}× | Ganho: {improvement:>5.1f}%')
    
    # Predição
    samples = data['predict'][dataset]['samples']
    baseline_pred = data['predict'][dataset]['baseline']
    optimized_pred = data['predict'][dataset]['optimized']
    
    print('  PREDIÇÃO:')
    for i, s in enumerate(samples):
        if i < len(baseline_pred) and i < len(optimized_pred):
            speedup = baseline_pred[i] / optimized_pred[i]
            improvement = ((baseline_pred[i] - optimized_pred[i]) / baseline_pred[i]) * 100
            print(f'    {s:>7} samples: {baseline_pred[i]:>8.2f}ms → {optimized_pred[i]:>8.2f}ms | '
                  f'Speedup: {speedup:.3f}× | Ganho: {improvement:>5.1f}%')

print('\n' + '='*80)
print('✅ Análise completa!')
