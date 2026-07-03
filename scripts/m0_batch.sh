#!/bin/bash
# Run M0 (reference preprocess + descriptors) for a list of ids in parallel.
# Usage: m0_batch.sh <idlist.txt> <NPAR>
idlist="$1"; npar="${2:-6}"
# Cap OpenMP/BLAS threads so parallel APBS/MSMS/TF processes don't oversubscribe 8 cores.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
echo "M0 batch start $(date '+%F %T'): $(wc -l < "$idlist") ids, -P $npar"
cat "$idlist" | xargs -P "$npar" -I{} bash /scratch/ymeng/masif-graph/scripts/m0_run_one.sh {}
echo "M0 batch end $(date '+%F %T')"
echo "=== M0_STATUS summary ==="
grep -h "M0_STATUS" /scratch/ymeng/masif-graph/logs/m0/*.log | sort | uniq -c | sed 's/^/  /'
