#!/bin/bash

echo "=== Iniciando Restauração dos Datasets ==="

# 1. OPTDIGITS (Optical Recognition of Handwritten Digits)
# O PDF menciona 1.797 amostras, que corresponde ao arquivo de teste do UCI.
echo "-> Baixando OptDigits..."
wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes -O optdigits.csv
# O arquivo original já é separado por vírgulas.
echo "   [OK] optdigits.csv restaurado."

# 2. SKIN SEGMENTATION
# O arquivo original usa TABs como separador, precisamos converter para CSV.
echo "-> Baixando Skin Segmentation..."
wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt
# Converte Tabs para Vírgulas e salva como csv
cat Skin_NonSkin.txt | tr -s '[:blank:]' ',' > skin_segmentation.csv
rm Skin_NonSkin.txt
echo "   [OK] skin_segmentation.csv restaurado."

# 3. COVERTYPE (Forest Cover Type)
# O arquivo vem compactado (.gz) e já é CSV.
echo "-> Baixando Covertype..."
wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz
gunzip -f covtype.data.gz
mv covtype.data covertype_dataset.csv
echo "   [OK] covertype_dataset.csv restaurado."

# 4. ADULT (Census Income)
# Incluído pois estava no seu script de execução anterior.
echo "-> Baixando Adult Dataset..."
wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
# Remove linhas em branco e espaços extras após vírgulas
grep -v "^$" adult.data | sed 's/, /,/g' > adult_dataset.csv
rm adult.data
echo "   [OK] adult_dataset.csv restaurado."

echo "=========================================="
echo "Todos os datasets foram recuperados!"
ls -lh *.csv