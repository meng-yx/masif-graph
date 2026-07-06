#!/bin/bash
# Phase-2 scale: preprocess NEW holo complexes (surfaces + descriptors) via the reference
# pipeline, so the probe can be enlarged. Runs niced (yields to floor work) and parallel.
# Skips ids already complete. Usage: holo_enlarge_batch.sh <idlist> <n_parallel> <max_ids>
set -u
idlist="$1"; NP="${2:-4}"; MAXN="${3:-80}"
REFDATA=/scratch/ymeng/masif-graph/masif-neosurf-af2/masif/data/masif_ppi_search
HERE=/scratch/ymeng/masif-graph
DESC="$REFDATA/descriptors/sc05/all_feat"

one() {
  id="$1"
  dd="$DESC/$id"
  if [ -s "$dd/p1_desc_straight.npy" ] && [ -s "$dd/p2_desc_flipped.npy" ]; then
    echo "SKIP $id"; return 0
  fi
  export TMPDIR="/tmp/p2holo_$id"; rm -rf "$TMPDIR"; mkdir -p "$TMPDIR"
  nice -n 19 bash "$HERE/scripts/m0_run_one.sh" "$id"
  rm -rf "$TMPDIR"
}
export -f one; export REFDATA HERE DESC

head -n "$MAXN" "$idlist" | \
  xargs -P "$NP" -I{} bash -c 'one "$@"' _ {}
echo "=== holo_enlarge_batch done $(date '+%F %T') ==="
