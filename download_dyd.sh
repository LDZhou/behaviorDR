#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=eecs
#SBATCH --time=8:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=download_dyd
#SBATCH --output=logs/download_dyd_%j.out
#SBATCH --error=logs/download_dyd_%j.err

echo "=== DYD Data Download Job Started: $(date) ==="

# 数据存放目录（在 hpc-share 上，空间大）
DATA_DIR=~/hpc-share/research/ecobee/dyd_data
mkdir -p ${DATA_DIR}/peak_climate
mkdir -p ${DATA_DIR}/ecoplus_sample
mkdir -p ${DATA_DIR}/dr_meta

# 激活 conda 环境
source ~/ENTER/etc/profile.d/conda.sh
conda activate ai_env

# Google Cloud 认证
export GOOGLE_APPLICATION_CREDENTIALS=~/hpc-share/research/ecobee/gcp-key.json

# 下载 peak_climate
echo "=== Downloading peak_climate: $(date) ==="
gsutil -m cp "gs://yish1/peak_climate_*.csv" ${DATA_DIR}/peak_climate/
echo "peak_climate done: $(date)"

# 下载 ecoplus_sample
echo "=== Downloading ecoplus_sample: $(date) ==="
gsutil -m cp "gs://yish1/ecoplus_sample_*.csv" ${DATA_DIR}/ecoplus_sample/
echo "ecoplus_sample done: $(date)"

# 下载 dr_meta
echo "=== Downloading dr_meta: $(date) ==="
gsutil -m cp "gs://yish1/dr_meta_*.csv" ${DATA_DIR}/dr_meta/
echo "dr_meta done: $(date)"

# 检查结果
echo ""
echo "=== Download Summary ==="
echo "peak_climate:"
ls -lh ${DATA_DIR}/peak_climate/ | tail -5
echo "File count: $(ls ${DATA_DIR}/peak_climate/*.csv 2>/dev/null | wc -l)"
echo ""
echo "ecoplus_sample:"
ls -lh ${DATA_DIR}/ecoplus_sample/ | tail -5
echo "File count: $(ls ${DATA_DIR}/ecoplus_sample/*.csv 2>/dev/null | wc -l)"
echo ""
echo "dr_meta:"
ls -lh ${DATA_DIR}/dr_meta/ | tail -5
echo "File count: $(ls ${DATA_DIR}/dr_meta/*.csv 2>/dev/null | wc -l)"
echo ""
echo "Total size:"
du -sh ${DATA_DIR}/*
du -sh ${DATA_DIR}

echo "=== Job Finished: $(date) ==="