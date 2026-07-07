#!/bin/bash
# M1: turn AF3 monomer models of a complex's two chains into surfaces+descriptors under a parallel
# id "{PDBID}AF_{C1}_{C2}", reusing the reference pipeline (mirrors Phase-2 repack_one.sh).
#
# Steps:
#   1. Relabel each chain's AF3 top-ranked model.cif -> holo-numbered PDB (masif_graph.af3.build_pdb).
#   2. Assemble both relabeled chains -> 00-raw_pdbs/{PDBID}AF.pdb.
#   3. Run reference 01-triangulate (per chain), 04-precompute (site + ppi_search),
#      compute_descriptors under id {PDBID}AF_{C1}_{C2}.
#
# Usage: af3_model_to_surface.sh 3TDM_A_B
set -u
id="$1"
PDBID=$(echo "$id" | cut -d_ -f1); C1=$(echo "$id" | cut -d_ -f2); C2=$(echo "$id" | cut -d_ -f3)
S="${2:-0}"; AFID="${PDBID}AS${S}"

REFROOT=/scratch/ymeng/masif-graph/masif-neosurf-af2
REFDATA=$REFROOT/masif/data/masif_ppi_search
SRC=$REFROOT/masif/source
SIF=$REFROOT/masif-neosurf_v0.1.sif
RAW=$REFDATA/data_preparation/00-raw_pdbs
MODELS=/work/upthomae/Meng/phase3_af3/models
PY=/work/upthomae/Meng/conda_envs/masif-graph/bin/python
LOGDIR=/scratch/ymeng/masif-graph/logs/phase3/af3_surf_samples
mkdir -p "$LOGDIR" "$RAW"
log="$LOGDIR/${id}.log"

export TMPDIR="/tmp/p3af_${AFID}"; rm -rf "$TMPDIR"; mkdir -p "$TMPDIR"

{
echo "=== AF3->SURF $id -> $AFID $(date '+%F %T') host=$(hostname) ==="

# --- 1. relabel each chain's top model ---
tmp=$(mktemp -d)
for C in "$C1" "$C2"; do
  name="${PDBID}_${C}"
  # top-ranked model.cif in the (timestamped) AF3 output dir
  cif=$(find "$MODELS/$name" -path "*seed-*_sample-${S}*" -name "*_model.cif" 2>/dev/null | head -1)
  if [ -z "$cif" ] || [ ! -s "$cif" ]; then echo "AF_STATUS $id FAIL no_cif_$C ($MODELS/$name)"; exit 2; fi
  PYTHONPATH=/scratch/ymeng/masif-graph/src "$PY" -m masif_graph.af3.build_pdb "$PDBID" "$C" "$cif" "$tmp/${C}_af.pdb" || { echo "AF_STATUS $id FAIL relabel_$C"; exit 3; }
  nat=$(grep -c '^ATOM' "$tmp/${C}_af.pdb"); echo "  relabel chain $C -> $nat atoms (from $cif)"
done

# --- 2. assemble the AF3 raw PDB (both chains) ---
rawout="$RAW/${AFID}.pdb"
{ grep -hE '^(ATOM|HETATM)' "$tmp/${C1}_af.pdb"; echo "TER";
  grep -hE '^(ATOM|HETATM)' "$tmp/${C2}_af.pdb"; echo "TER"; echo "END"; } > "$rawout"
echo "  assembled $rawout ($(grep -c '^ATOM' "$rawout") atoms)"
rm -rf "$tmp"

# --- 3. reference pipeline under the AF id ---
export PYTHONPATH="${PYTHONPATH:-}:$SRC:$REFDATA"
SEXEC="singularity exec --bind $REFROOT:$REFROOT $SIF python"
cd "$REFDATA" || { echo "AF_STATUS $id FAIL cd"; exit 4; }

echo "--- 01 triangulate $AFID $C1 ---"; timeout 1800 $SEXEC "$SRC/data_preparation/01-pdb_extract_and_triangulate.py" "${AFID}_${C1}"; echo "  rc=$?"
echo "--- 01 triangulate $AFID $C2 ---"; timeout 1800 $SEXEC "$SRC/data_preparation/01-pdb_extract_and_triangulate.py" "${AFID}_${C2}"; echo "  rc=$?"
echo "--- 04 precompute masif_site ---"; timeout 1800 $SEXEC "$SRC/data_preparation/04-masif_precompute.py" masif_site "${AFID}_${C1}_${C2}"; echo "  rc=$?"
echo "--- 04 precompute masif_ppi_search ---"; timeout 1800 $SEXEC "$SRC/data_preparation/04-masif_precompute.py" masif_ppi_search "${AFID}_${C1}_${C2}"; echo "  rc=$?"
echo "--- descriptors ---"; timeout 1800 $SEXEC "$SRC/masif_ppi_search/masif_ppi_search_comp_desc.py" nn_models.sc05.all_feat.custom_params "${AFID}_${C1}_${C2}"; echo "  rc=$?"

# --- verify ---
dd="$REFDATA/descriptors/sc05/all_feat/${AFID}_${C1}_${C2}"
pc="$REFDATA/data_preparation/04b-precomputation_12A/precomputation/${AFID}_${C1}_${C2}"
ok=1
for f in "$dd/p1_desc_straight.npy" "$dd/p1_desc_flipped.npy" "$dd/p2_desc_straight.npy" "$dd/p2_desc_flipped.npy" \
         "$pc/p1_X.npy" "$pc/p2_X.npy" "$pc/p1_iface_labels.npy" "$pc/p2_iface_labels.npy"; do
  [ -s "$f" ] || { echo "MISSING: $f"; ok=0; }
done
echo "=== END $id $(date '+%F %T') ==="
if [ "$ok" = 1 ]; then echo "AF_STATUS $id OK"; else echo "AF_STATUS $id FAIL missing_outputs"; fi
} >"$log" 2>&1
tail -1 "$log"
