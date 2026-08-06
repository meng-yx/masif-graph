#!/bin/bash
# Holo per-complex reference preprocessing (reconstructed for Phase 5): PDBID_C1_C2 -> surfaces +
# descriptors via the .sif. Mirrors scripts/af3_model_to_surface.sh but for the holo (experimental)
# structure. The reference 00-pdb_download uses a dead wwPDB FTP endpoint (-> 126-byte empty stub),
# so step 00 is patched to fetch directly from RCSB then protonate exactly as 00 did.
# Resumable: SKIPs if descriptors already exist. Usage: m0_run_one.sh PDBID_C1_C2
set -u
id="$1"
PDBID=$(echo "$id" | cut -d_ -f1); C1=$(echo "$id" | cut -d_ -f2); C2=$(echo "$id" | cut -d_ -f3)

REFROOT=/scratch/ymeng/masif-graph/masif-neosurf-af2
REFDATA=$REFROOT/masif/data/masif_ppi_search
SRC=$REFROOT/masif/source
SIF=$REFROOT/masif-neosurf_v0.1.sif
DD=$REFDATA/descriptors/sc05/all_feat
RAW=$REFDATA/data_preparation/00-raw_pdbs
LOGDIR=/scratch/ymeng/masif-graph/logs/phase5/holo_surf
mkdir -p "$LOGDIR" "$RAW"
log="$LOGDIR/${id}.log"

[ -s "$DD/$id/p1_desc_straight.npy" ] && { echo "SKIP $id"; exit 0; }

export TMPDIR="/tmp/p5holo_${id}"; rm -rf "$TMPDIR"; mkdir -p "$TMPDIR"
export PYTHONPATH="${PYTHONPATH:-}:$SRC:$REFDATA"
SEXEC="singularity exec --bind $REFROOT:$REFROOT --bind /work:/work $SIF python"
cd "$REFDATA" || { echo "M0_STATUS $id FAIL cd"; exit 4; }

{
echo "=== HOLO->SURF $id $(date '+%F %T') host=$(hostname) ==="
echo "--- 00 fetch+protonate $id ---"
curl -sf -o "$TMPDIR/${PDBID}.pdb" "https://files.rcsb.org/download/${PDBID}.pdb"
natom=$(grep -c '^ATOM' "$TMPDIR/${PDBID}.pdb" 2>/dev/null || echo 0)
echo "  fetched ${PDBID}.pdb atoms=$natom"
if [ "$natom" -lt 10 ]; then
  echo "  rc=1 (fetch failed)"; echo "M0_STATUS $id FAIL fetch"
else
  $SEXEC -c "import sys; sys.path.insert(0,'$SRC'); from input_output.protonate import protonate; protonate('$TMPDIR/${PDBID}.pdb', '$RAW/${PDBID}.pdb')"; echo "  rc=$?"
  echo "--- 01 triangulate ${PDBID}_${C1} ---"; timeout 1800 $SEXEC "$SRC/data_preparation/01-pdb_extract_and_triangulate.py" "${PDBID}_${C1}"; echo "  rc=$?"
  echo "--- 01 triangulate ${PDBID}_${C2} ---"; timeout 1800 $SEXEC "$SRC/data_preparation/01-pdb_extract_and_triangulate.py" "${PDBID}_${C2}"; echo "  rc=$?"
  echo "--- 04 precompute masif_site ---";       timeout 1800 $SEXEC "$SRC/data_preparation/04-masif_precompute.py" masif_site "$id";        echo "  rc=$?"
  echo "--- 04 precompute masif_ppi_search ---"; timeout 1800 $SEXEC "$SRC/data_preparation/04-masif_precompute.py" masif_ppi_search "$id";  echo "  rc=$?"
  echo "--- descriptors ---";                    timeout 1800 $SEXEC "$SRC/masif_ppi_search/masif_ppi_search_comp_desc.py" nn_models.sc05.all_feat.custom_params "$id"; echo "  rc=$?"
  if [ -s "$DD/$id/p1_desc_straight.npy" ]; then echo "M0_STATUS $id OK"; else echo "M0_STATUS $id FAIL"; fi
fi
rm -rf "$TMPDIR"
} >> "$log" 2>&1
grep "M0_STATUS $id" "$log" | tail -1
