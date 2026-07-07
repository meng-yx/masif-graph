#!/bin/bash
# Wait for M3 training data, then submit the real pilot sweep (72 train / 30 eval). Idempotent-ish.
set -u
cd /scratch/ymeng/masif-graph
i=0
until [ -f logs/M3_TRAIN_DATA_READY ]; do i=$((i+1)); [ $i -ge 360 ] && { echo "wait cap"; break; }; sleep 60; done
echo "M3PILOT: training data ready $(date '+%T')"
# usable train/eval counts
export PYTHONPATH=/scratch/ymeng/masif-graph/src
/work/upthomae/Meng/conda_envs/atomsurf_h100/bin/python -c "
from masif_graph.m3.dataset import usable_complexes
import os
tr=usable_complexes('logs/phase3/m3_data',[l.strip() for l in open('logs/phase3/m3_train_ids.txt')])
tra=[c for c in tr if os.path.exists(f'logs/phase3/m3_data/{c}__af3__p1.npz')]
print(f'M3PILOT: train usable={len(tr)} with_af3={len(tra)}')"
# submit the sweep: (reg, inv, graph)
sub(){ sbatch --export=ALL,TAG=$1,STEPS=1500,EVALEVERY=100,LR=5e-4,INVW=$3,REGW=$2,${4:+NOGRAPH=1} scripts/m3_train.sbatch 2>&1 | grep -oP 'job \K[0-9]+'; }
echo "pilot reg2_inv1_graph  -> $(sub p_r2i1g   2.0 1.0)"
echo "pilot reg2_inv3_graph  -> $(sub p_r2i3g   2.0 3.0)"
echo "pilot reg1_inv2_graph  -> $(sub p_r1i2g   1.0 2.0)"
echo "pilot reg2_inv1_NOgraph-> $(sub p_r2i1n   2.0 1.0 1)"
echo "M3PILOT_SWEEP_SUBMITTED $(date '+%T')"
