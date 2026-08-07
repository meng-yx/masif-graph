#!/bin/bash
# Phase-6 C(c).3 axis 3 — add the FROZEN MaSIF descriptors to the 14 neosurface systems, so the
# benchmark can score the published representation on the identical patches.
# The main neosurface build deliberately skips the descriptor net (the learned encoder never reads
# it), and deletes the 04b precompute; both are regenerated here. The .ply surfaces survive, so
# step 01 is not repeated.
# Submit: sbatch --array=0-13 scripts/p6C_neosurf_desc.sbatch <out_npz_dir>
#SBATCH --job-name=p6C_nbdesc
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=27G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase6C/neosurf/desc_%A_%a.out
set -u
export HOME=/home/ymeng OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
OUT="${1:-/work/upthomae/Meng/phase6C/npz_neosurf}"
ROOT=/scratch/ymeng/masif-graph
REFROOT=$ROOT/masif-neosurf-af2; REFDATA=$REFROOT/masif/data/masif_ppi_search
SRC=$REFROOT/masif/source; SIF=$REFROOT/masif-neosurf_v0.1.sif
BENCH=$REFROOT/computational_benchmark/benchmark_pdbs.txt
PY=/work/upthomae/Meng/conda_envs/masif-graph/bin/python
export TMPDIR="$ROOT/tmp/nbd_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
LINE=$(grep -v '^#' "$BENCH" | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
PDB=$(echo "$LINE"|cut -d, -f1); C1=$(echo "$LINE"|cut -d, -f2); C2=$(echo "$LINE"|cut -d, -f3)
SYS="nb${PDB}"; CID="${SYS}_${C1}_${C2}"
echo "host=$(hostname) sys=$CID start=$(date)"
export PYTHONPATH="$SRC:$REFDATA"
SEXEC="singularity exec --bind $REFROOT:$REFROOT --bind /work:/work --bind $ROOT:$ROOT --bind $TMPDIR:$TMPDIR $SIF python"
cd "$REFDATA" || exit 1
PC=data_preparation/04b-precomputation_12A/precomputation/$CID
[ -s "$PC/p2_input_feat.npy" ] || timeout 3600 $SEXEC "$SRC/data_preparation/04-masif_precompute.py" \
   masif_ppi_search "$CID" > "$TMPDIR/04.log" 2>&1
[ -s "$PC/p2_input_feat.npy" ] || { echo "$PDB FAIL_04"; exit 1; }
timeout 3600 $SEXEC "$SRC/masif_ppi_search/masif_ppi_search_comp_desc.py" \
   nn_models.sc05.all_feat.custom_params "$CID" > "$TMPDIR/desc.log" 2>&1
DD=descriptors/sc05/all_feat/$CID
[ -s "$DD/p2_desc_straight.npy" ] || { echo "$PDB FAIL_DESC $(grep -oE '[A-Za-z]*Error.*' "$TMPDIR/desc.log"|tail -1)"; exit 1; }
REP=$(PYTHONPATH=$ROOT/src $PY -m masif_graph.p6.neosurf --bench "$BENCH" --only "$PDB" --out "$OUT" 2>&1 | grep '^{' | tail -1)
rm -rf "$PC"
echo "$PDB REBUILT $REP"
