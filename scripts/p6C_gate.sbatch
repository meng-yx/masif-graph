#!/bin/bash
# Phase-6 C(c).3 — the three evaluation axes for one checkpoint. ALWAYS on a compute node
# (the Phase-5 gate OOM-kills the login node).
# Usage: sbatch scripts/p6C_gate.sbatch <tag> <ckpt> <npz_eval_dir>
#SBATCH --job-name=p6C_gate
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase6C/gate_%x_%j.out
set -u
export HOME=/home/ymeng
export PATH="/home/ymeng/miniconda3/bin:/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=16
source /home/ymeng/miniconda3/etc/profile.d/conda.sh
conda activate masif-graph
cd /scratch/ymeng/masif-graph
TAG="${1:?tag}"; CKPT="${2:?ckpt}"; EVALDIR="${3:-/work/upthomae/Meng/phase6C/npz_eval}"
P6=/work/upthomae/Meng/phase6C
OUT=logs/phase6C/results; mkdir -p "$OUT"

echo "=== AXIS 1: do-no-harm PPI gate (Phase-5 287-clean, 4-cell) tag=$TAG $(date) ==="
for PK in pos pos_sc; do
  python -u -m masif_graph.p5.retrieval_bench --data "$EVALDIR" \
    --ids logs/phase5/eval_sc304_clean_vs_enc.txt --ckpt "$CKPT" --center --pos-key $PK \
    --out "$OUT/gate_${TAG}_${PK}.json"
done

echo "=== AXIS 2: mixed held-out retrieval (PPI holdout + protein-ligand holdout) $(date) ==="
python -u -m masif_graph.p6.mixed_bench \
  --ppi-data "$P6/npz_ppi" --ppi-ids "$P6/split/val_ppi_stageB.txt" \
  --pl-data "$P6/npz_pl" --pl-ids "$P6/split/val_pl.txt" \
  --ckpt "$CKPT" --center --pos-key pos --out "$OUT/mixed_${TAG}.json"
echo "--- axis 2b: scaffold-unseen subset (clean on BOTH leakage axes) ---"
python -u -m masif_graph.p6.mixed_bench \
  --pl-data "$P6/npz_pl" --pl-ids "$P6/split/val_pl_scaffold_unseen.txt" \
  --ckpt "$CKPT" --center --pos-key pos --out "$OUT/mixed_scafclean_${TAG}.json"

echo "=== AXIS 3: neosurface benchmark (28 ligand-induced cases) $(date) ==="
python -u -m masif_graph.p6.neosurf_bench --neosurf-data "$P6/npz_neosurf" \
  --decoy-data "$P6/npz_eval" --decoy-ids logs/phase5/eval_sc304_clean_vs_enc.txt \
  --ckpt "$CKPT" --center --lig-radius 6.0 --max-db-atoms 1500 \
  --out "$OUT/neosurf_${TAG}.json"
echo "=== GATE DONE $TAG $(date) ==="
