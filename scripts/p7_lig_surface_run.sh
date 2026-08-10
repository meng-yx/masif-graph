#!/bin/bash
# Phase-7 — drive p7_lig_surface.py inside the .sif over a chunk of PDBbind ids.
# Usage: p7_lig_surface_run.sh <ids_file> <out_dir> [apbs_mode]
set -u
IDS="$1"; OUT="$2"; MODE="${3:-selfpqr}"
ROOT=/scratch/ymeng/masif-graph
REFROOT=$ROOT/masif-neosurf-af2; REFDATA=$REFROOT/masif/data/masif_ppi_search
SRC=$REFROOT/masif/source; SIF=$REFROOT/masif-neosurf_v0.1.sif
export LC_ALL=C.UTF-8 LANG=C.UTF-8
export TMPDIR="${TMPDIR:-/tmp}/p7lig_$$"; mkdir -p "$TMPDIR" "$OUT"
trap 'rm -rf "$TMPDIR"' EXIT
cd "$REFDATA" || exit 1
PYTHONPATH="$SRC:$REFDATA" singularity exec \
  --bind $REFROOT:$REFROOT --bind /work:/work --bind $ROOT:$ROOT --bind "$TMPDIR:$TMPDIR" \
  "$SIF" python $ROOT/scripts/p7_lig_surface.py \
  --ids-file "$IDS" --out-dir "$OUT" --apbs-mode "$MODE"
