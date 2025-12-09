import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Diretório com os logs
results_dir = 'results'

# Estrutura para armazenar dados
data = []

# Padrão do nome do arquivo: {version}_{dataset}_{samples}_perf_*.log
filename_pattern = r'(baseline|optimized)_(\w+)_([\d.]+)k_perf_'

# Padrões para extrair métricas
patterns = {
    'l1_loads': r'([\d.]+)\s+L1-dcache-load\s',
    'l1_misses': r'([\d.]+)\s+L1-dcache-load-misses',
    'l2_accesses': r'([\d.]+)\s+l2_cache_accesses_from_dc_misses',
    'l2_hits': r'([\d.]+)\s+l2_cache_hits_from_dc_misses',
    'l2_misses': r'([\d.]+)\s+l2_cache_misses_from_dc_misses',
    'branch_loads': r'([\d.]+)\s+branch-load\s',
    'branch_misses': r'([\d.]+)\s+branch-load-misses',
    'time_elapsed': r'([\d,]+)\s+seconds time elapsed'
}

# Processar todos os arquivos *perf*.log
for filename in os.listdir(results_dir):
    if 'perf' in filename and filename.endswith('.log'):
        filepath = os.path.join(results_dir, filename)
        
        # Extrair informações do nome do arquivo
        match = re.search(filename_pattern, filename)
        if not match:
            continue
        
        version, dataset, samples = match.groups()
        samples_float = float(samples)
        
        # Ler conteúdo do arquivo
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extrair métricas
        metrics = {
            'version': version,
            'dataset': dataset,
            'samples': samples_float
        }
        
        for metric_name, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                value = match.group(1).replace('.', '').replace(',', '.')
                try:
                    metrics[metric_name] = float(value)
                except:
                    metrics[metric_name] = 0
            else:
                metrics[metric_name] = 0
        
        # Calcular taxas
        if metrics['l1_loads'] > 0:
            metrics['l1_miss_rate'] = (metrics['l1_misses'] / metrics['l1_loads']) * 100
        else:
            metrics['l1_miss_rate'] = 0
            
        if metrics['l2_accesses'] > 0:
            metrics['l2_hit_rate'] = (metrics['l2_hits'] / metrics['l2_accesses']) * 100
            metrics['l2_miss_rate'] = (metrics['l2_misses'] / metrics['l2_accesses']) * 100
        else:
            metrics['l2_hit_rate'] = 0
            metrics['l2_miss_rate'] = 0
            
        if metrics['branch_loads'] > 0:
            metrics['branch_miss_rate'] = (metrics['branch_misses'] / metrics['branch_loads']) * 100
        else:
            metrics['branch_miss_rate'] = 0
        
        data.append(metrics)

# Criar DataFrame
df = pd.DataFrame(data)
df = df.sort_values(['dataset', 'samples', 'version'])

print("="*100)
print("📊 ANÁLISE DE PERFORMANCE - BASELINE vs OTIMIZADO")
print("="*100)

# Agrupar por dataset
datasets = df['dataset'].unique()

for dataset in datasets:
    df_dataset = df[df['dataset'] == dataset]
    samples_list = sorted(df_dataset['samples'].unique())
    
    print(f"\n{'='*100}")
    print(f"📁 DATASET: {dataset.upper()}")
    print(f"{'='*100}")
    
    for sample_size in samples_list:
        df_sample = df_dataset[df_dataset['samples'] == sample_size]
        baseline = df_sample[df_sample['version'] == 'baseline']
        optimized = df_sample[df_sample['version'] == 'optimized']
        
        if baseline.empty or optimized.empty:
            continue
        
        b = baseline.iloc[0]
        o = optimized.iloc[0]
        
        print(f"\n📈 Tamanho: {sample_size}k samples")
        print("-"*100)
        
        # Tempo
        time_speedup = b['time_elapsed'] / o['time_elapsed'] if o['time_elapsed'] > 0 else 0
        time_improvement = ((b['time_elapsed'] - o['time_elapsed']) / b['time_elapsed'] * 100) if b['time_elapsed'] > 0 else 0
        print(f"⏱️  TEMPO:")
        print(f"   Baseline:  {b['time_elapsed']:.3f}s")
        print(f"   Otimizado: {o['time_elapsed']:.3f}s")
        print(f"   → Speedup: {time_speedup:.2f}× ({time_improvement:+.1f}%)")
        
        # L1 Cache
        l1_reduction = ((b['l1_loads'] - o['l1_loads']) / b['l1_loads'] * 100) if b['l1_loads'] > 0 else 0
        print(f"\n🔵 L1 CACHE:")
        print(f"   Acessos Baseline:  {b['l1_loads']:>15,.0f}  (Miss Rate: {b['l1_miss_rate']:.2f}%)")
        print(f"   Acessos Otimizado: {o['l1_loads']:>15,.0f}  (Miss Rate: {o['l1_miss_rate']:.2f}%)")
        print(f"   → Redução de acessos: {l1_reduction:+.1f}%")
        
        # L2 Cache
        l2_reduction = ((b['l2_accesses'] - o['l2_accesses']) / b['l2_accesses'] * 100) if b['l2_accesses'] > 0 else 0
        print(f"\n🟡 L2 CACHE:")
        print(f"   Acessos Baseline:  {b['l2_accesses']:>15,.0f}  (Hit Rate: {b['l2_hit_rate']:.2f}%)")
        print(f"   Acessos Otimizado: {o['l2_accesses']:>15,.0f}  (Hit Rate: {o['l2_hit_rate']:.2f}%)")
        print(f"   → Redução de acessos: {l2_reduction:+.1f}%")
        
        # Branch Prediction
        branch_reduction = ((b['branch_misses'] - o['branch_misses']) / b['branch_misses'] * 100) if b['branch_misses'] > 0 else 0
        print(f"\n🔀 BRANCH PREDICTION:")
        print(f"   Misses Baseline:  {b['branch_misses']:>15,.0f}  (Miss Rate: {b['branch_miss_rate']:.2f}%)")
        print(f"   Misses Otimizado: {o['branch_misses']:>15,.0f}  (Miss Rate: {o['branch_miss_rate']:.2f}%)")
        print(f"   → Redução de misses: {branch_reduction:+.1f}%")

print(f"\n{'='*100}")
print("✅ Análise completa!")
print(f"{'='*100}\n")

# Gerar gráficos comparativos - SEPARADOS
metrics_to_plot = [
    ('time_elapsed', 'Tempo de Execução (s) - MENOR É MELHOR ⬇', 'Tempo', False),
    ('l1_miss_rate', 'L1 Cache Miss Rate (%) - MENOR É MELHOR ⬇', 'L1 Miss Rate', True),
    ('l2_miss_rate', 'L2 Cache Miss Rate (%) - MENOR É MELHOR ⬇', 'L2 Miss Rate', True),
    ('branch_miss_rate', 'Branch Miss Rate (%) - MENOR É MELHOR ⬇', 'Branch Miss Rate', True)
]

for metric, ylabel, title, is_percentage in metrics_to_plot:
    for dataset in sorted(datasets):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        df_dataset = df[df['dataset'] == dataset]
        baseline_data = df_dataset[df_dataset['version'] == 'baseline'].sort_values('samples')
        optimized_data = df_dataset[df_dataset['version'] == 'optimized'].sort_values('samples')
        
        x_labels = [f"{s:.1f}k" for s in baseline_data['samples']]
        x_pos = range(len(x_labels))
        width = 0.35
        
        bars1 = ax.bar([p - width/2 for p in x_pos], baseline_data[metric], width, 
                       label='Baseline', color='#FF0000', alpha=0.8)
        bars2 = ax.bar([p + width/2 for p in x_pos], optimized_data[metric], width, 
                       label='Otimizado', color='#00AA00', alpha=0.8)
        
        ax.set_xlabel('Número de Amostras', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', color='darkgreen')
        ax.set_title(f'{title} - {dataset.capitalize()}', 
                     fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores em cima das barras
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}{"%" if is_percentage else "s"}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}{"%" if is_percentage else "s"}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'imgs/perf_{metric}_{dataset}.png', dpi=300, bbox_inches='tight')
        print(f'✅ Gráfico salvo: imgs/perf_{metric}_{dataset}.png')
        plt.close()

print('\n📊 Total: 12 gráficos de performance gerados (4 métricas × 3 datasets)')
