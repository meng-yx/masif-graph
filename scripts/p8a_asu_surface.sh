#!/bin/bash
# Phase-8 A4 — reference surfaces + descriptors for ONE asymmetric-unit chain pair, so mined
# bio/crystal interfaces can be scored with the Stage-1 encoder (which needs vert_feat, and
# io.reference.load_complex additionally requires the 80-D MaSIF descriptors).
#
# Modelled on scripts/repack_one.sh, but the raw PDB comes from the Phase-8A RCSB cache rather
# than the reference tree: the reference 00-raw_pdbs cache holds the EVAL set, and A4 deliberately
# probes TRAINING entries.
#
# Usage: p8a_asu_surface.sh <PDBID> <c1> <c2>
set -u
PDBID="$1"; C1="$2"; C2="$3"
id="${PDBID}_${C1}_${C2}"

REFROOT=/scratch/ymeng/masif-graph/masif-neosurf-af2
REFDATA=$REFROOT/masif/data/masif_ppi_search
SRC=$REFROOT/masif/source
SIF=$REFROOT/masif-neosurf_v0.1.sif
RAW=$REFDATA/data_preparation/00-raw_pdbs
CACHE=/work/upthomae/Meng/phase8A/a4_pdbs
LOGDIR=/scratch/ymeng/masif-graph/logs/phase8A/a4/surf
mkdir -p "$LOGDIR" "$RAW"
log="$LOGDIR/${id}.log"

dd="$REFDATA/descriptors/sc05/all_feat/${id}"
pc="$REFDATA/data_preparation/04b-precomputation_12A/precomputation/${id}"
if [ -s "$dd/p1_desc_straight.npy" ] && [ -s "$dd/p2_desc_flipped.npy" ]; then
  echo "ASU_STATUS $id SKIP already_built"; exit 0
fi

export TMPDIR="/tmp/p8a_${id}"
rm -rf "$TMPDIR"; mkdir -p "$TMPDIR"

{
echo "=== ASU SURFACE $id $(date '+%F %T') host=$(hostname) ==="
if [ ! -s "$RAW/${PDBID}.pdb" ]; then
  [ -s "$CACHE/${PDBID}.pdb" ] || { echo "ASU_STATUS $id FAIL no_raw_pdb"; exit 2; }
  cp "$CACHE/${PDBID}.pdb" "$RAW/${PDBID}.pdb"
fi

export PYTHONPATH="${PYTHONPATH:-}:$SRC:$REFDATA"
SEXEC="singularity exec --bind $REFROOT:$REFROOT --bind /work:/work $SIF python"
cd "$REFDATA" || { echo "ASU_STATUS $id FAIL cd"; exit 3; }

for C in "$C1" "$C2"; do
  if [ ! -s "data_preparation/01-benchmark_surfaces/${PDBID}_${C}.ply" ]; then
    echo "--- 01 triangulate ${PDBID}_${C} ---"
    timeout 1800 $SEXEC "$SRC/data_preparation/01-pdb_extract_and_triangulate.py" "${PDBID}_${C}"
    echo "  rc=$?"
  else
    echo "--- 01 ${PDBID}_${C} cached ---"
  fi
done
echo "--- 04 masif_site ---";       timeout 1800 $SEXEC "$SRC/data_preparation/04-masif_precompute.py" masif_site "$id"; echo "  rc=$?"
echo "--- 04 masif_ppi_search ---"; timeout 1800 $SEXEC "$SRC/data_preparation/04-masif_precompute.py" masif_ppi_search "$id"; echo "  rc=$?"
echo "--- descriptors ---";         timeout 1800 $SEXEC "$SRC/masif_ppi_search/masif_ppi_search_comp_desc.py" nn_models.sc05.all_feat.custom_params "$id"; echo "  rc=$?"

ok=1
for f in "$dd/p1_desc_straight.npy" "$dd/p2_desc_flipped.npy" "$pc/p1_X.npy" "$pc/p2_X.npy" \
         "$pc/p1_iface_labels.npy" "$pc/p2_iface_labels.npy"; do
  [ -s "$f" ] || { echo "MISSING: $f"; ok=0; }
done
echo "=== END $id $(date '+%F %T') ==="
if [ "$ok" = 1 ]; then echo "ASU_STATUS $id OK"; else echo "ASU_STATUS $id FAIL missing_outputs"; fi
} >"$log" 2>&1
rm -rf "$TMPDIR"
tail -1 "$log"
