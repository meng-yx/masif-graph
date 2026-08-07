#!/bin/bash
# Phase-6 C(b).0 — 14-D -> 26-D re-featurization of the surviving PPI npz, as a SLURM array.
# Submit: sbatch --array=0-<K-1>%8 scripts/p6C_refeat_array.sbatch <chunkdir> <srcdir> <dstdir> <state>
# Kept to a modest concurrency: each task pulls missing entries from files.rcsb.org.
#SBATCH --job-name=p6C_refeat
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase6C/refeat/%A_%a.out

set -u
export HOME=/home/ymeng
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CHUNKDIR="${1:-/scratch/ymeng/masif-graph/logs/phase6C/refeat_chunks}"
SRC="${2:-/work/upthomae/Meng/phase4/stageA_full_npz}"
DST="${3:-/work/upthomae/Meng/phase6C/npz_ppi}"
STATE="${4:-holo}"
ROOT=/scratch/ymeng/masif-graph
CHUNK=$(printf '%s/chunk_%04d.txt' "$CHUNKDIR" "$SLURM_ARRAY_TASK_ID")
PY=/work/upthomae/Meng/conda_envs/masif-graph/bin/python
mkdir -p "$DST"
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID chunk=$CHUNK src=$SRC dst=$DST state=$STATE start=$(date)"
PYTHONPATH=$ROOT/src $PY -m masif_graph.p6.refeat --ids "$CHUNK" --src "$SRC" --dst "$DST" \
  --state "$STATE" \
  --pdb-cache $ROOT/data/rcsb_cache \
  --chain-dir $ROOT/masif-neosurf-af2/masif/data/masif_ppi_search/data_preparation/01-benchmark_pdbs
echo "=== refeat task $SLURM_ARRAY_TASK_ID DONE $(date) ==="
