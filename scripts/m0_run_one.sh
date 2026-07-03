#!/bin/bash
# M0: run reference preprocessing + descriptor computation for ONE complex id.
# Usage: m0_run_one.sh PDBID_C1_C2
# Runs from masif/data/masif_ppi_search/ so masif_opts relative paths (incl. model_data)
# resolve correctly. Logs to logs/m0/<id>.log. Emits a final status line the caller greps.
id="$1"
REFDATA=/scratch/ymeng/masif-graph/masif-neosurf-af2/masif/data/masif_ppi_search
LOGDIR=/scratch/ymeng/masif-graph/logs/m0
mkdir -p "$LOGDIR"
log="$LOGDIR/$id.log"

pdbid=$(echo "$id" | cut -d_ -f1)
c1=$(echo "$id" | cut -d_ -f2)
c2=$(echo "$id" | cut -d_ -f3)

cd "$REFDATA" || { echo "M0_STATUS $id FAIL cd"; exit 99; }

{
  echo "=== M0 $id START $(date '+%F %T') host=$(hostname) cwd=$(pwd) ==="
  echo "--- [1/2] data_prepare_one.sh ---"
  timeout 3600 ./data_prepare_one.sh "$id"
  echo "data_prepare rc=$?"
  echo "--- [2/2] compute_descriptors.sh ---"
  timeout 3600 ./compute_descriptors.sh "$id"
  echo "compute_descriptors rc=$?"
  echo "=== M0 $id END $(date '+%F %T') ==="
} >"$log" 2>&1

# Verify the artifacts we actually need exist and are non-empty.
ply1="$REFDATA/data_preparation/01-benchmark_surfaces/${pdbid}_${c1}.ply"
ply2="$REFDATA/data_preparation/01-benchmark_surfaces/${pdbid}_${c2}.ply"
pdb1="$REFDATA/data_preparation/01-benchmark_pdbs/${pdbid}_${c1}.pdb"
pdb2="$REFDATA/data_preparation/01-benchmark_pdbs/${pdbid}_${c2}.pdb"
dd="$REFDATA/descriptors/sc05/all_feat/$id"
pc="$REFDATA/data_preparation/04b-precomputation_12A/precomputation/$id"

ok=1
for f in "$ply1" "$ply2" "$pdb1" "$pdb2" \
         "$dd/p1_desc_straight.npy" "$dd/p1_desc_flipped.npy" \
         "$dd/p2_desc_straight.npy" "$dd/p2_desc_flipped.npy" \
         "$pc/p1_X.npy" "$pc/p2_X.npy" "$pc/p1_iface_labels.npy" "$pc/p2_iface_labels.npy"; do
  if [ ! -s "$f" ]; then echo "MISSING: $f" >>"$log"; ok=0; fi
done

if [ "$ok" = 1 ]; then
  echo "M0_STATUS $id OK" | tee -a "$log"
else
  echo "M0_STATUS $id FAIL missing_outputs" | tee -a "$log"
fi
