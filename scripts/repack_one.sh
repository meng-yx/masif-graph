#!/bin/bash
# Phase-2 M0: generate the apo-like (fixed-backbone monomer repack) surfaces+descriptors for
# one complex, under a parallel id "{PDBID}RP_{C1}_{C2}", reusing the reference pipeline.
#
# Steps:
#   1. FASPR-repack each extracted holo chain 01-benchmark_pdbs/{PDBID}_{Ck}.pdb IN ISOLATION
#      (unbound monomer context, backbone fixed).
#   2. Assemble both repacked chains -> 00-raw_pdbs/{PDBID}RP.pdb (still in the holo frame).
#   3. Run reference 01-triangulate (per chain), 04-precompute (site + ppi_search),
#      compute_descriptors under id {PDBID}RP_{C1}_{C2}.  (00-download is skipped.)
#
# Usage: repack_one.sh 3TDM_A_B
set -u
id="$1"
PDBID=$(echo "$id" | cut -d_ -f1)
C1=$(echo "$id" | cut -d_ -f2)
C2=$(echo "$id" | cut -d_ -f3)
RPID="${PDBID}RP"

REFROOT=/scratch/ymeng/masif-graph/masif-neosurf-af2
REFDATA=$REFROOT/masif/data/masif_ppi_search
SRC=$REFROOT/masif/source
SIF=$REFROOT/masif-neosurf_v0.1.sif
FASPR_DIR=/work/upthomae/Meng/tools/FASPR
FASPR=$FASPR_DIR/FASPR
BENCH=$REFDATA/data_preparation/01-benchmark_pdbs
RAW=$REFDATA/data_preparation/00-raw_pdbs
LOGDIR=/scratch/ymeng/masif-graph/logs/p2_repack
mkdir -p "$LOGDIR" "$RAW"
log="$LOGDIR/${id}.log"

# Per-job TMPDIR so concurrent repacks don't collide on MSMS/APBS temp files (/tmp is bound
# into the singularity container by default).
export TMPDIR="/tmp/p2rp_${RPID}"
rm -rf "$TMPDIR"; mkdir -p "$TMPDIR"

{
echo "=== REPACK $id -> $RPID $(date '+%F %T') host=$(hostname) ==="

# --- 1. FASPR repack each chain in isolation ---
tmp=$(mktemp -d)
for C in "$C1" "$C2"; do
  inpdb="$BENCH/${PDBID}_${C}.pdb"
  if [ ! -s "$inpdb" ]; then echo "REPACK_STATUS $id FAIL missing_input $inpdb"; exit 2; fi
  ( cd "$FASPR_DIR" && "$FASPR" -i "$inpdb" -o "$tmp/${C}_rp.pdb" ) || { echo "REPACK_STATUS $id FAIL faspr_$C"; exit 3; }
  nat=$(grep -c '^ATOM' "$tmp/${C}_rp.pdb")
  echo "  FASPR chain $C -> $nat atoms"
done

# --- 2. assemble the repacked raw PDB (both chains, holo frame) ---
rawout="$RAW/${RPID}.pdb"
{
  grep -hE '^(ATOM|HETATM)' "$tmp/${C1}_rp.pdb"
  echo "TER"
  grep -hE '^(ATOM|HETATM)' "$tmp/${C2}_rp.pdb"
  echo "TER"
  echo "END"
} > "$rawout"
echo "  assembled $rawout ($(grep -c '^ATOM' "$rawout") atoms)"
rm -rf "$tmp"

# --- 3. reference pipeline under the RP id ---
export PYTHONPATH="${PYTHONPATH:-}:$SRC:$REFDATA"
BIND="$REFROOT:$REFROOT"
SEXEC="singularity exec --bind $BIND $SIF python"
cd "$REFDATA" || { echo "REPACK_STATUS $id FAIL cd"; exit 4; }

echo "--- 01 triangulate $RPID $C1 ---"; timeout 1800 $SEXEC "$SRC/data_preparation/01-pdb_extract_and_triangulate.py" "${RPID}_${C1}"; echo "  rc=$?"
echo "--- 01 triangulate $RPID $C2 ---"; timeout 1800 $SEXEC "$SRC/data_preparation/01-pdb_extract_and_triangulate.py" "${RPID}_${C2}"; echo "  rc=$?"
echo "--- 04 precompute masif_site ---"; timeout 1800 $SEXEC "$SRC/data_preparation/04-masif_precompute.py" masif_site "${RPID}_${C1}_${C2}"; echo "  rc=$?"
echo "--- 04 precompute masif_ppi_search ---"; timeout 1800 $SEXEC "$SRC/data_preparation/04-masif_precompute.py" masif_ppi_search "${RPID}_${C1}_${C2}"; echo "  rc=$?"
echo "--- descriptors ---"; timeout 1800 $SEXEC "$SRC/masif_ppi_search/masif_ppi_search_comp_desc.py" nn_models.sc05.all_feat.custom_params "${RPID}_${C1}_${C2}"; echo "  rc=$?"

# --- verify ---
dd="$REFDATA/descriptors/sc05/all_feat/${RPID}_${C1}_${C2}"
pc="$REFDATA/data_preparation/04b-precomputation_12A/precomputation/${RPID}_${C1}_${C2}"
ok=1
for f in "$dd/p1_desc_straight.npy" "$dd/p1_desc_flipped.npy" "$dd/p2_desc_straight.npy" "$dd/p2_desc_flipped.npy" \
         "$pc/p1_X.npy" "$pc/p2_X.npy" "$pc/p1_iface_labels.npy" "$pc/p2_iface_labels.npy"; do
  [ -s "$f" ] || { echo "MISSING: $f"; ok=0; }
done
echo "=== END $id $(date '+%F %T') ==="
if [ "$ok" = 1 ]; then echo "REPACK_STATUS $id OK"; else echo "REPACK_STATUS $id FAIL missing_outputs"; fi
} >"$log" 2>&1
tail -1 "$log"
