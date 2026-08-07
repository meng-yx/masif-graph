#!/bin/bash
# Phase-6 C(c).3 axis 3 — surface build for ONE ligand-induced ternary benchmark system.
# Usage: p6C_neosurf_one.sh <PDB> <chains1> <chains2> <out_npz_dir>
# Builds both subunit surfaces (Path B: the drug is NOT in the surface) + the pairwise 04b
# precompute, then the npz via masif_graph.p6.neosurf. Idempotent on the contacts npz.
set -u
PDB="$1"; C1="$2"; C2="$3"; OUT="$4"
ROOT=/scratch/ymeng/masif-graph
REFROOT=$ROOT/masif-neosurf-af2; REFDATA=$REFROOT/masif/data/masif_ppi_search
SRC=$REFROOT/masif/source; SIF=$REFROOT/masif-neosurf_v0.1.sif
RAW=$REFDATA/data_preparation/00-raw_pdbs
PY=${PY:-/work/upthomae/Meng/conda_envs/masif-graph/bin/python}
SYS="nb${PDB}"

[ -f "$OUT/${SYS}__ligcontacts.npz" ] && { echo "$PDB DONE(cached)"; exit 0; }
export TMPDIR="${TMPDIR:-/tmp}/nb_$$_$PDB"; mkdir -p "$TMPDIR" "$RAW" "$OUT"
trap 'rm -rf "$TMPDIR"' EXIT

if [ ! -s "$RAW/${SYS}.pdb" ]; then
  curl -sS --max-time 120 -o "$RAW/${SYS}.pdb" "https://files.rcsb.org/download/${PDB}.pdb" \
    || { echo "$PDB FAIL_DOWNLOAD"; exit 1; }
fi

export PYTHONPATH="$SRC:$REFDATA"
SEXEC="singularity exec --bind $REFROOT:$REFROOT --bind /work:/work --bind $ROOT:$ROOT --bind $TMPDIR:$TMPDIR $SIF python"
cd "$REFDATA" || exit 1
for CH in "$C1" "$C2"; do
  if [ ! -s "data_preparation/01-benchmark_surfaces/${SYS}_${CH}.ply" ]; then
    timeout 3600 $SEXEC "$SRC/data_preparation/01-pdb_extract_and_triangulate.py" "${SYS}_${CH}" \
      > "$TMPDIR/01_${CH}.log" 2>&1
    [ -s "data_preparation/01-benchmark_surfaces/${SYS}_${CH}.ply" ] || {
      echo "$PDB FAIL_01_${CH} $(grep -oE '[A-Za-z]*Error.*' "$TMPDIR/01_${CH}.log" | tail -1)"; exit 1; }
  fi
done
timeout 3600 $SEXEC "$SRC/data_preparation/04-masif_precompute.py" masif_ppi_search "${SYS}_${C1}_${C2}" \
  > "$TMPDIR/04.log" 2>&1
PC=data_preparation/04b-precomputation_12A/precomputation/${SYS}_${C1}_${C2}
[ -s "$PC/p2_input_feat.npy" ] || { echo "$PDB FAIL_04 $(grep -oE '[A-Za-z]*Error.*' "$TMPDIR/04.log" | tail -1)"; exit 1; }

REP=$(PYTHONPATH=$ROOT/src $PY -m masif_graph.p6.neosurf \
        --bench "$REFROOT/computational_benchmark/benchmark_pdbs.txt" --only "$PDB" \
        --out "$OUT" 2>&1 | grep '^{' | tail -1)
rm -rf "$PC"
echo "$PDB $REP"
case "$REP" in *'"ok": true'*) exit 0 ;; *) exit 1 ;; esac
