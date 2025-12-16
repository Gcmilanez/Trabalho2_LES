import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
INPUT_FOLDER = './resultsPredict' 
OUTPUT_ROOT = 'resultados_graficos_finais'
FILENAME_PATTERN = re.compile(r"(baseline|optimized)_([a-zA-Z0-9_]+)_([\d\.,]+[kK]?)")

METRICS_MAP = {
    r"L1-dcache-load-misses": "L1 Misses",
    r"l2_cache_hits_from_dc_misses": "L2 Hits",
    r"l2_cache_misses_from_dc_misses": "L2 Misses",
    r"branch-load-misses": "Branch Misses"
}

# ==============================================================================
# FUNÇÕES DE LEITURA
# ==============================================================================

def parse_value(line):
    raw_num = line.strip().split()[0]
    clean_num = raw_num.replace('.', '')
    try:
        return int(clean_num)
    except ValueError:
        return 0

def parse_time(line):
    match = re.search(r"([\d\.,]+)\s+seconds time elapsed", line)
    if match:
        raw_time = match.group(1)
        clean_time = raw_time.replace('.', '').replace(',', '.')
        return float(clean_time)
    return None

def get_metadata(filename):
    match = FILENAME_PATTERN.search(filename)
    if match:
        method = match.group(1).capitalize()
        dataset = match.group(2)
        sample_str = match.group(3)
        
        multiplier = 1
        val_str = sample_str.lower().replace('k', '').replace(',', '.')
        if 'k' in sample_str.lower():
            multiplier = 1000
            
        try:
            sample_val = int(float(val_str) * multiplier)
        except:
            sample_val = 0
            
        return method, dataset, sample_val, sample_str
    return None, None, 0, filename

# ==============================================================================
# LEITURA E PROCESSAMENTO
# ==============================================================================

records = []
print(f"Lendo logs da pasta: {os.path.abspath(INPUT_FOLDER)}...")

files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.log') or f.endswith('.txt')]

if not files:
    print("❌ Nenhum arquivo encontrado!")
    exit()

for file in files:
    method, dataset, sample_val, sample_label = get_metadata(file)
    if not method: continue
        
    metrics = {name: 0 for name in METRICS_MAP.values()}
    metrics['Time'] = 0.0
    
    with open(os.path.join(INPUT_FOLDER, file), 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            for pattern, alias in METRICS_MAP.items():
                if pattern in line:
                    metrics[alias] = parse_value(line)
            if "seconds time elapsed" in line:
                t = parse_time(line)
                if t is not None:
                    metrics['Time'] = t
    
    record = {'Dataset': dataset, 'Method': method, 'Samples': sample_val, 'Label': sample_label, **metrics}
    records.append(record)

df = pd.DataFrame(records)
df = df.sort_values(by=['Dataset', 'Samples', 'Method'])

# ==============================================================================
# GERAÇÃO DOS GRÁFICOS
# ==============================================================================

datasets = df['Dataset'].unique()
metrics_to_plot = list(METRICS_MAP.values()) + ['Time']

print(f"\nGerando gráficos em: {os.path.abspath(OUTPUT_ROOT)}")

for dataset in datasets:
    dataset_dir = os.path.join(OUTPUT_ROOT, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    
    ds_data = df[df['Dataset'] == dataset]
    sample_sizes = ds_data['Samples'].unique()
    sample_sizes.sort()
    
    x_labels = []
    for s in sample_sizes:
        label = ds_data[ds_data['Samples'] == s]['Label'].iloc[0]
        x_labels.append(label)

    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(9, 7))
        
        x = np.arange(len(sample_sizes))
        width = 0.35
        
        baseline_vals = []
        optimized_vals = []
        
        for s in sample_sizes:
            b_row = ds_data[(ds_data['Samples'] == s) & (ds_data['Method'] == 'Baseline')]
            val_b = b_row[metric].values[0] if not b_row.empty else 0
            baseline_vals.append(val_b)
            
            o_row = ds_data[(ds_data['Samples'] == s) & (ds_data['Method'] == 'Optimized')]
            val_o = o_row[metric].values[0] if not o_row.empty else 0
            optimized_vals.append(val_o)
            
        ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#d9534f', alpha=0.9, edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, optimized_vals, width, label='Optimized', color='#5cb85c', alpha=0.9, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Tamanho da Amostra', fontsize=12)
        ax.set_ylabel(metric + " (Escala Log)", fontsize=12)
        ax.set_title(f'{dataset} - {metric}', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=11)
        ax.legend(fontsize=10)
        
        ax.grid(True, which="major", ls="-", alpha=0.4, color='gray')
        ax.grid(True, which="minor", ls=":", alpha=0.2, color='gray')

        # === CORREÇÃO DA ESCALA LOGARÍTMICA ===
        all_vals = baseline_vals + optimized_vals
        max_val = max(all_vals) if all_vals else 0
        min_pos_val = min([v for v in all_vals if v > 0]) if any(v > 0 for v in all_vals) else 1e-3

        if max_val > 0:
            ax.set_yscale('log')
            
            # Limites Inteligentes: Do menor valor positivo (ou perto) até uma ordem acima do máximo
            ax.set_ylim(bottom=min_pos_val * 0.5, top=max_val * 50)
            
            # Localizador de Ticks Logarítmicos (Potências de 10)
            locmaj = ticker.LogLocator(base=10.0, numticks=15)
            ax.yaxis.set_major_locator(locmaj)
            
            # Formatador Híbrido: Científico para pequenos, K/M/B para grandes
            def smart_log_formatter(x, pos):
                if x == 0: return "0"
                # Para números muito pequenos (Tempo < 0.1s)
                if x < 0.1: return f'{x:.3g}' 
                if x < 1: return f'{x:.2g}'
                # Para números normais
                if x < 1000: return f'{x:.0f}'
                # Para números grandes
                if x >= 1e9: return f'{x*1e-9:.0f}B'
                if x >= 1e6: return f'{x*1e-6:.0f}M'
                if x >= 1e3: return f'{x*1e-3:.0f}K'
                return f'{x:.0f}'

            ax.yaxis.set_major_formatter(ticker.FuncFormatter(smart_log_formatter))
        
        plt.tight_layout()
        
        filename = f"{metric.replace(' ', '_')}.png"
        save_path = os.path.join(dataset_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    print(f"✅ Dataset '{dataset}': Gráficos salvos em {dataset_dir}")

print("\nProcesso concluído!")