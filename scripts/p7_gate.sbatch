#!/bin/bash
# Phase-7 — three-axis evaluation for one checkpoint, on ITS OWN representation.
# Usage: sbatch scripts/p7_gate.sbatch <tag> <ckpt> <pl_data> <neosurf_data> [composite_data]
# A Phase-6C model must be scored on Phase-6C ligand graphs and a Phase-7 model on Phase-7 ones;
# crossing them would be an out-of-distribution eval, not a comparison.
#SBATCH --job-name=p7_gate
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase7/gate_%x_%j.out
export HOME=/home/ymeng LC_ALL=C.UTF-8
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=/work/upthomae/Meng/conda_envs/masif-graph/bin/python
ROOT=/scratch/ymeng/masif-graph
export PYTHONPATH=$ROOT/src
cd $ROOT
TAG="${1:?tag}"; CKPT="${2:?ckpt}"
PLD="${3:-/work/upthomae/Meng/phase7/npz_pl}"
NBD="${4:-/work/upthomae/Meng/phase7/npz_neosurf}"
CMP="${5:-}"   # empty = skip the composite arm (Phase-6C models never saw ligand vertices)
P7=/work/upthomae/Meng/phase7
OUT=logs/phase7/results; mkdir -p "$OUT"

echo "=== TAG=$TAG CKPT=$CKPT PL=$PLD NB=$NBD CMP=$CMP $(date) ==="
echo "=== AXIS 1: do-no-harm PPI gate (unchanged 287-clean eval) ==="
for PK in pos pos_sc; do
  $PY -u -m masif_graph.p5.retrieval_bench --data /work/upthomae/Meng/phase6C/npz_eval \
    --ids logs/phase5/eval_sc304_clean_vs_enc.txt --ckpt "$CKPT" --center --pos-key $PK \
    --out "$OUT/gate_${TAG}_${PK}.json"
done

echo "=== AXIS 2: mixed held-out (+ TRAIN-set, the capacity diagnostic) ==="
$PY -u -m masif_graph.p6.mixed_bench --ppi-data $P7/npz_ppi --ppi-ids $P7/split/val_ppi_stageB.txt \
  --pl-data "$PLD" --pl-ids $P7/split/val_pl.txt --ckpt "$CKPT" --center --pos-key pos \
  --out "$OUT/mixed_${TAG}.json"
$PY -u -m masif_graph.p6.mixed_bench --pl-data "$PLD" --pl-ids $P7/split/val_pl_scaffold_unseen.txt \
  --ckpt "$CKPT" --center --pos-key pos --out "$OUT/mixed_scafclean_${TAG}.json"
# the Phase-6C gate question: can it fit the ligand axis on data it TRAINED on? same set sizes.
head -197 $P7/split/train_ppi.txt > /tmp/p7_tr_ppi_$$.txt
head -300 $P7/split/train_pl.txt  > /tmp/p7_tr_pl_$$.txt
$PY -u -m masif_graph.p6.mixed_bench --ppi-data $P7/npz_ppi --ppi-ids /tmp/p7_tr_ppi_$$.txt \
  --pl-data "$PLD" --pl-ids /tmp/p7_tr_pl_$$.txt --ckpt "$CKPT" --center --pos-key pos \
  --out "$OUT/mixedtrain_${TAG}.json"
rm -f /tmp/p7_tr_ppi_$$.txt /tmp/p7_tr_pl_$$.txt

echo "=== AXIS 3: neosurface (separate-surface + composite queries) ==="
CMPARG=""; [ -n "$CMP" ] && [ -d "$CMP" ] && CMPARG="--composite-data $CMP"
$PY -u -m masif_graph.p6.neosurf_bench --neosurf-data "$NBD" $CMPARG \
  --decoy-data /work/upthomae/Meng/phase6C/npz_eval \
  --decoy-ids logs/phase5/eval_sc304_clean_vs_enc.txt \
  --ckpt "$CKPT" --center --lig-radius 6.0 --max-db-atoms 1500 \
  --out "$OUT/neosurf_${TAG}.json"
echo "=== P7 GATE DONE $TAG $(date) ==="
