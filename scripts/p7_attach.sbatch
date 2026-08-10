#!/bin/bash
# Phase-7 — attach ligand surfaces to the Phase-6C ligand npz for the whole corpus.
#SBATCH --job-name=p7_attach
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase7/attach_%j.out
export HOME=/home/ymeng LC_ALL=C.UTF-8
export OMP_NUM_THREADS=4
PY=/work/upthomae/Meng/conda_envs/masif-graph/bin/python
ROOT=/scratch/ymeng/masif-graph
cd $ROOT
PYTHONPATH=$ROOT/src $PY -m masif_graph.p7.lig_vertices \
  --ids $ROOT/logs/phase7/all_pl_ids.txt \
  --src /work/upthomae/Meng/phase6C/npz_pl \
  --dst /work/upthomae/Meng/phase7/npz_pl \
  --surf-dir /work/upthomae/Meng/phase7/lig_surf --copy-protein
echo "=== ATTACH DONE $(date) ==="
