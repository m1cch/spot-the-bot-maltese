#!/bin/bash
# Финальный прогон на лучшей конфигурации (SVD, n=4, m=16).
# Старые отчёты n4_m8 переименовываются в *_n4m8 как baseline.
set -e
cd ~/spot-the-bot-maltese
PY=.venv/bin/python
SVD=embeddings/svd/mt_svd_k1024.npz

mv -f results/clustering/wishart_report.json    results/clustering/wishart_report_n4m8.json    2>/dev/null || true
mv -f results/clustering/wishart_sizes.png      results/clustering/wishart_sizes_n4m8.png      2>/dev/null || true
mv -f results/topology/topology_report.json     results/topology/topology_report_n4m8.json     2>/dev/null || true
mv -f results/topology/persistence_diagrams.png results/topology/persistence_diagrams_n4m8.png 2>/dev/null || true

echo "### build n4_m16 datasets ###"
$PY -u scripts/build_dataset.py --corpus corpus_clean_L2/all_clean.shuf.txt --dict $SVD -n 4 -m 16 --name human --max-rows 300000
$PY -u scripts/build_dataset.py --corpus corpus_bot/all_clean.shuf.txt       --dict $SVD -n 4 -m 16 --name bot   --max-rows 300000

echo "### wishart n4_m16 ###"
$PY -u scripts/wishart_compare.py --human datasets/human__n4_m16.npy --bot datasets/bot__n4_m16.npy -k 11 --hh 1 --sample 30000

echo "### topology n4_m16 ###"
$PY -u scripts/topology.py --human datasets/human__n4_m16.npy --bot datasets/bot__n4_m16.npy --points 2000 --runs 5

echo ALL_FINAL_DONE
