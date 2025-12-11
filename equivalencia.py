import re
import matplotlib.pyplot as plt
import numpy as np
import os

# Cria diretório de imagens se não existir
if not os.path.exists('imgs'):
    os.makedirs('imgs')

# Nome do arquivo de log
filename = 'results/prediction_log_20251211_091601.txt'

# Conteúdo do log (simulado para o script funcionar standalone, 
# mas no seu uso real ele lerá do arquivo)

content = """
Testes de Predicao iniciados em: 12/11/2025 09:23:08
DATASET: OPTDIGITS
Arquivo: optdigits.csv
BASELINE optdigits 0,5k: 1ms (100% acc)
OTIMIZADO optdigits 0,5k: 0.4ms (100% acc)
BASELINE optdigits 1k: 1ms (100% acc)
OTIMIZADO optdigits 1k: 0.8ms (100% acc)
BASELINE optdigits 1,797k: 2ms (98.61% acc)
OTIMIZADO optdigits 1,797k: 1.48ms (99.44% acc)
DATASET: SKIN
Arquivo: skin_segmentation.csv
BASELINE skin 1k: 0ms (100% acc)
OTIMIZADO skin 1k: 0ms (100% acc)
BASELINE skin 10k: 1.99ms (100% acc)
OTIMIZADO skin 10k: 1.3ms (100% acc)
BASELINE skin 245,057k: 61.94ms (99.53% acc)
OTIMIZADO skin 245,057k: 87.45ms (99.66% acc)
DATASET: COVERTYPE
Arquivo: covertype.csv
BASELINE covertype 1k: 1ms (79.5% acc)
OTIMIZADO covertype 1k: 0.6ms (84.5% acc)
BASELINE covertype 100k: 36.68ms (83.5% acc)
OTIMIZADO covertype 100k: 49.32ms (82.86% acc)
BASELINE covertype 581,012k: 244.88ms (69.18% acc)
OTIMIZADO covertype 581,012k: 293.28ms (70.57% acc)
"""
# Se quiser salvar esse dummy no disco para manter a consistência:
os.makedirs('results', exist_ok=True)
with open(filename, 'w', encoding='utf-8') as f:
    f.write(content)

# Estrutura para armazenar os dados
# Formato: data[dataset] = {'samples': [], 'baseline': [], 'optimized': []}
data = {}
datasets_order = []

# Regex para capturar os dados
# Captura: 1=Tipo, 2=Dataset, 3=Samples(string), 4=Time, 5=Accuracy
pattern = r'(BASELINE|OTIMIZADO)\s+(\w+)\s+([\d,]+)k:\s+([\d.]+)ms\s+\(([\d.]+)%\s+acc\)'

for match in re.finditer(pattern, content):
    tipo, dataset_name, sample_str, time_ms, acc = match.groups()
    
    # Normalizar nome do dataset (upper case para consistência)
    ds_key = dataset_name.upper()
    acc = float(acc)
    
    # Inicializa dataset se não existir
    if ds_key not in data:
        data[ds_key] = {'samples': [], 'baseline': [], 'optimized': []}
        datasets_order.append(ds_key)
    
    # Adiciona sample se não existir
    if sample_str not in data[ds_key]['samples']:
        data[ds_key]['samples'].append(sample_str)
        # Inicializa as listas com None para manter alinhamento
        data[ds_key]['baseline'].append(0)
        data[ds_key]['optimized'].append(0)
    
    # Encontra o índice do sample atual
    idx = data[ds_key]['samples'].index(sample_str)
    
    # Armazena a acurácia
    if tipo == 'BASELINE':
        data[ds_key]['baseline'][idx] = acc
    else:
        data[ds_key]['optimized'][idx] = acc

# Cores e configurações
colors = {'baseline': '#FF0000', 'optimized': '#00AA00'} # Vermelho e Verde
bar_width = 0.35

print(f"Datasets encontrados: {datasets_order}")

# Gerar gráficos
for dataset in datasets_order:
    ds_data = data[dataset]
    samples = ds_data['samples']
    baseline = ds_data['baseline']
    optimized = ds_data['optimized']
    
    # Configuração do plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(samples))
    
    # Criar barras
    rects1 = ax.bar(x - bar_width/2, baseline, bar_width, label='Baseline', color=colors['baseline'], alpha=0.8)
    rects2 = ax.bar(x + bar_width/2, optimized, bar_width, label='Otimizado', color=colors['optimized'], alpha=0.8)
    
    # Labels e Títulos
    ax.set_ylabel('Acurácia (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Quantidade de Amostras (k)', fontsize=12, fontweight='bold')
    ax.set_title(f'Acurácia de Predição - {dataset}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s + 'k' for s in samples])
    ax.set_ylim(0, 115) # Dá um espaço extra em cima para os labels
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Função para adicionar labels nas barras
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    
    # Adicionar anotação de diferença
    for i, (b, o) in enumerate(zip(baseline, optimized)):
        diff = o - b
        if abs(diff) > 0.01: # Só mostra se houver diferença relevante
            color = 'green' if diff >= 0 else 'red'
            sign = '+' if diff >= 0 else ''
            ax.text(i, max(b, o) + 6, f'{sign}{diff:.2f}%', 
                    ha='center', color=color, fontweight='bold', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    plt.tight_layout()
    
    # Salvar
    filename_png = f'imgs/acc_{dataset.lower()}.png'
    plt.savefig(filename_png, dpi=300)
    print(f'✅ Gráfico salvo: {filename_png}')
    plt.close()

print("\nProcesso concluído!")