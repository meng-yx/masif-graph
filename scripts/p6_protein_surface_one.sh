#!/bin/bash
# Normal (protein-only) surface+descriptors for a PDBbind protein.pdb via the .sif. Usage: <id> <chain>
set -u
ID="$1"; CH="${2:-A}"
REFROOT=/scratch/ymeng/masif-graph/masif-neosurf-af2; REFDATA=$REFROOT/masif/data/masif_ppi_search
SRC=$REFROOT/masif/source; SIF=$REFROOT/masif-neosurf_v0.1.sif
RAW=$REFDATA/data_preparation/00-raw_pdbs; DD=$REFDATA/descriptors/sc05/all_feat
PLB=/scratch/ymeng/masif-graph/data/pdbbind/$ID
LOG=/scratch/ymeng/masif-graph/logs/phase6/protsurf_${ID}.log; mkdir -p "$RAW" logs/phase6
PID="pl${ID}"   # synthetic pdb id to avoid clashes with RCSB ids
cp "$PLB/${ID}_protein.pdb" "$RAW/${PID}.pdb"
export PYTHONPATH="${PYTHONPATH:-}:$SRC:$REFDATA"
SEXEC="singularity exec --bind $REFROOT:$REFROOT --bind /work:/work --bind /scratch/ymeng/masif-graph:/scratch/ymeng/masif-graph $SIF python"
cd "$REFDATA"
{ echo "=== PROTSURF $ID ($PID)_$CH $(date '+%T') ==="
  echo "--- 01 ---"; timeout 1800 $SEXEC "$SRC/data_preparation/01-pdb_extract_and_triangulate.py" "${PID}_${CH}"; echo " rc=$?"
  echo "--- 04 site ---"; timeout 1800 $SEXEC "$SRC/data_preparation/04-masif_precompute.py" masif_site "${PID}_${CH}_${CH}"; echo " rc=$?"
  echo "--- 04 ppi ---"; timeout 1800 $SEXEC "$SRC/data_preparation/04-masif_precompute.py" masif_ppi_search "${PID}_${CH}_${CH}"; echo " rc=$?"
  echo "--- desc ---"; timeout 1800 $SEXEC "$SRC/masif_ppi_search/masif_ppi_search_comp_desc.py" nn_models.sc05.all_feat.custom_params "${PID}_${CH}_${CH}"; echo " rc=$?"
  ls data_preparation/01-benchmark_surfaces/${PID}_${CH}.ply 2>/dev/null && echo "PLY_OK" || echo "PLY_MISSING"
} >> "$LOG" 2>&1
grep -E 'rc=|PLY_|Error|Traceback' "$LOG" | tail -8
