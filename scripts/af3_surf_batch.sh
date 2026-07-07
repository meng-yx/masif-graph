#!/bin/bash
# Run af3_model_to_surface for a list of complex ids in parallel. Skips ids whose AF3 descriptors
# already exist. Usage: af3_surf_batch.sh <idlist> <n_parallel>
set -u
idlist="$1"; NP="${2:-6}"
REFDATA=/scratch/ymeng/masif-graph/masif-neosurf-af2/masif/data/masif_ppi_search
HERE=/scratch/ymeng/masif-graph
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
run_one(){
  id="$1"; pdb=$(echo "$id"|cut -d_ -f1); c1=$(echo "$id"|cut -d_ -f2); c2=$(echo "$id"|cut -d_ -f3)
  dd="$REFDATA/descriptors/sc05/all_feat/${pdb}AF_${c1}_${c2}"
  if [ -s "$dd/p1_desc_straight.npy" ] && [ -s "$dd/p2_desc_flipped.npy" ]; then echo "SKIP $id"; return 0; fi
  bash "$HERE/scripts/af3_model_to_surface.sh" "$id"
}
export -f run_one; export REFDATA HERE
grep -E '^[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+$' "$idlist" | xargs -P "$NP" -I{} bash -c 'run_one "$@"' _ {}
echo "=== af3_surf_batch done $(date '+%F %T') ==="
grep -h "AF_STATUS" /scratch/ymeng/masif-graph/logs/phase3/af3_surf/*.log 2>/dev/null | awk '{print $3}' | sort | uniq -c
