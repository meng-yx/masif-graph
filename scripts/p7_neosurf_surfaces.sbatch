#!/bin/bash
# Phase-7 S3 — for all 14 ternary systems: the drug's own surface + BOTH subunits' composite
# protein+drug surfaces. Small (14 systems x 3 builds), so one job rather than an array.
#SBATCH --job-name=p7_neosurf
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=13G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase7/neosurf_%j.out
set -u
export HOME=/home/ymeng LC_ALL=C.UTF-8 LANG=C.UTF-8
export OMP_NUM_THREADS=2
ROOT=/scratch/ymeng/masif-graph
REFROOT=$ROOT/masif-neosurf-af2; REFDATA=$REFROOT/masif/data/masif_ppi_search
SRC=$REFROOT/masif/source; SIF=$REFROOT/masif-neosurf_v0.1.sif
BENCH=$REFROOT/computational_benchmark/benchmark_pdbs.txt
OUT=/work/upthomae/Meng/phase7/neosurf
export TMPDIR="$ROOT/tmp/p7nb_${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR" "$OUT"
trap 'rm -rf "$TMPDIR"' EXIT
SEXEC="singularity exec --bind $REFROOT:$REFROOT --bind /work:/work --bind $ROOT:$ROOT --bind $TMPDIR:$TMPDIR $SIF python"
cd "$REFDATA" || exit 1
export PYTHONPATH="$SRC:$REFDATA"

grep -v '^#' "$BENCH" | while IFS=, read -r PDB C1 C2 DRUG SDFNAME; do
  [ -z "$PDB" ] && continue
  ASYM=$(echo "$SDFNAME" | cut -d_ -f2)
  SDF=$ROOT/data/neosurf_sdf/${PDB}_${ASYM}.sdf
  [ -s "$SDF" ] || { echo "$PDB MISSING_SDF $SDF"; continue; }
  # (a) the drug's own surface -> attaches to nb<PDB>__lig.npz
  if [ ! -s "$OUT/lignb${PDB}_feat.npy" ]; then
    $SEXEC $ROOT/scripts/p7_lig_surface.py --sdf "$SDF" \
      --out-prefix "$OUT/lignb${PDB}" --id "nb${PDB}" --apbs-mode selfpqr 2>&1 | grep '^{'
  fi
  # (b) composite protein+drug surface for BOTH subunits (D7-4: inference only)
  for CH in "$C1" "$C2"; do
    PDBF=$REFDATA/data_preparation/01-benchmark_pdbs/nb${PDB}_${CH}.pdb
    [ -s "$PDBF" ] || { echo "$PDB MISSING_PDB $PDBF"; continue; }
    [ -s "$OUT/cmp_nb${PDB}_${CH}_feat.npy" ] && continue
    $SEXEC $ROOT/scripts/p7_composite_surface.py --pdb "$PDBF" --sdf "$SDF" \
      --out-prefix "$OUT/cmp_nb${PDB}_${CH}" --id "${PDB}_${CH}" 2>&1 | grep '^{'
  done
done
echo "=== P7 NEOSURF SURFACES DONE $(date) ==="
