#!/bin/bash

# Cria diretório de resultados se não existir
mkdir -p resultsPredict

# ============================================================
# OPTDIGITS (500, 1000, 1797)
# ============================================================

# 500 Amostras (0.5k)
perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/baseline_optdigits_0.5k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_baseline_predict optdigits.csv models/baseline_optdigits_0.5k.model 500 5

perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/optimized_optdigits_0.5k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_optimized_predict optdigits.csv models/optimized_optdigits_0.5k.model 500 5

# 1000 Amostras (1.0k)
perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/baseline_optdigits_1.0k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_baseline_predict optdigits.csv models/baseline_optdigits_1.0k.model 1000 5

perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/optimized_optdigits_1.0k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_optimized_predict optdigits.csv models/optimized_optdigits_1.0k.model 1000 5

# 1797 Amostras (1.8k)
perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/baseline_optdigits_1.8k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_baseline_predict optdigits.csv models/baseline_optdigits_1.8k.model 1797 5

perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/optimized_optdigits_1.8k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_optimized_predict optdigits.csv models/optimized_optdigits_1.8k.model 1797 5


# ============================================================
# ADULT DATASET (1000, 10000, 45222)
# ============================================================

# 1000 Amostras (1.0k)
perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/baseline_adult_1.0k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_baseline_predict adult_dataset.csv models/baseline_adult_1.0k.model 1000 5

perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/optimized_adult_1.0k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_optimized_predict adult_dataset.csv models/optimized_adult_1.0k.model 1000 5

# 10000 Amostras (10.0k)
perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/baseline_adult_10.0k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_baseline_predict adult_dataset.csv models/baseline_adult_10.0k.model 10000 5

perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/optimized_adult_10.0k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_optimized_predict adult_dataset.csv models/optimized_adult_10.0k.model 10000 5

# 45222 Amostras (45.0k)
perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/baseline_adult_45.0k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_baseline_predict adult_dataset.csv models/baseline_adult_45.0k.model 45222 5

perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/optimized_adult_45.0k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_optimized_predict adult_dataset.csv models/optimized_adult_45.0k.model 45222 5


# ============================================================
# SKIN SEGMENTATION (1000, 10000, 245057)
# ============================================================

# 1000 Amostras (1.0k)
perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/baseline_skin_1.0k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_baseline_predict skin_segmentation.csv models/baseline_skin_1.0k.model 1000 5

perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/optimized_skin_1.0k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_optimized_predict skin_segmentation.csv models/optimized_skin_1.0k.model 1000 5

# 10000 Amostras (10.0k)
perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/baseline_skin_10.0k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_baseline_predict skin_segmentation.csv models/baseline_skin_10.0k.model 10000 5

perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/optimized_skin_10.0k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_optimized_predict skin_segmentation.csv models/optimized_skin_10.0k.model 10000 5

# 245057 Amostras (245.1k)
perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/baseline_skin_245.1k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_baseline_predict skin_segmentation.csv models/baseline_skin_245.1k.model 245057 5

perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/optimized_skin_245.1k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_optimized_predict skin_segmentation.csv models/optimized_skin_245.1k.model 245057 5


# ============================================================
# COVERTYPE (Substituindo Arrhythmia)
# ============================================================

# 10k
perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/baseline_covertype_10.0k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_baseline_predict covertype_dataset.csv models/baseline_covertype_10.0k.model 10000 5

perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/optimized_covertype_10.0k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_optimized_predict covertype_dataset.csv models/optimized_covertype_10.0k.model 10000 5

# 100k
perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/baseline_covertype_100.0k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_baseline_predict covertype_dataset.csv models/baseline_covertype_100.0k.model 100000 5

perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/optimized_covertype_100.0k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_optimized_predict covertype_dataset.csv models/optimized_covertype_100.0k.model 100000 5

# 581k (Full)
perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/baseline_covertype_581.0k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_baseline_predict covertype_dataset.csv models/baseline_covertype_581.0k.model 581012 5

perf stat -e L1-dcache-load,L1-dcache-load-misses,l2_cache_accesses_from_dc_misses,l2_cache_hits_from_dc_misses,l2_cache_misses_from_dc_misses,l3_cache_accesses,l3_misses,branch-load,branch-load-misses \
-o resultsPredict/optimized_covertype_581.0k_predict_perf_$(date +%Y%m%d_%H%M%S).log \
-- ./forest_optimized_predict covertype_dataset.csv models/optimized_covertype_581.0k.model 581012 5


