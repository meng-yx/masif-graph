#!/bin/bash
# Phase-6 C(b).1 — full Path-B preprocessing of ONE PDBbind complex.
#   prep protein pdb (ligand-contacting chains -> one pseudo-chain A)
#   -> .sif 01 (MSMS + APBS + .ply)  -> .sif 04b (masif_ppi_search precompute, p1 only)
#   -> protein + ligand npz + contacts  -> delete the 40 MB precompute dir
# The descriptor net and 04a/masif_site are NOT run: the learned encoder never reads the frozen
# MaSIF descriptors, so they are pure cost. Idempotent: exits early if the contacts npz exists.
# Usage: p6C_pdbbind_one.sh <pdbbind_id> <out_npz_dir>
set -u
ID="$1"; OUT="$2"
ROOT=/scratch/ymeng/masif-graph
REFROOT=$ROOT/masif-neosurf-af2; REFDATA=$REFROOT/masif/data/masif_ppi_search
SRC=$REFROOT/masif/source; SIF=$REFROOT/masif-neosurf_v0.1.sif
RAW=$REFDATA/data_preparation/00-raw_pdbs
PRECOMP=$REFDATA/data_preparation/04b-precomputation_12A/precomputation
PY=${PY:-/work/upthomae/Meng/conda_envs/masif-graph/bin/python}
PID="pl${ID}"

[ -f "$OUT/${PID}__contacts.npz" ] && { echo "$ID DONE(cached)"; exit 0; }

# per-task scratch so concurrent array tasks never share an MSMS/APBS temp file
export TMPDIR="${TMPDIR:-/tmp}/p6c_$$_$ID"; mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
mkdir -p "$RAW" "$OUT"

PREP=$(PYTHONPATH=$ROOT/src $PY -c "
import json, sys
from masif_graph.p6.pl_graph import prep_protein_pdb
print(json.dumps(prep_protein_pdb('$ROOT/data/pdbbind', '$ID', '$RAW/${PID}.pdb')))
" 2>&1 | tail -1)
case "$PREP" in
  *'"skip"'*) echo "$ID SKIP $PREP"; exit 0 ;;
  *'"written"'*) : ;;
  *) echo "$ID PREPFAIL $PREP"; exit 1 ;;
esac

export PYTHONPATH="$SRC:$REFDATA"
SEXEC="singularity exec --bind $REFROOT:$REFROOT --bind /work:/work --bind $ROOT:$ROOT --bind $TMPDIR:$TMPDIR $SIF python"
cd "$REFDATA" || exit 1
timeout 3600 $SEXEC "$SRC/data_preparation/01-pdb_extract_and_triangulate.py" "${PID}_A" > "$TMPDIR/01.log" 2>&1
rc1=$?
if [ ! -s "$REFDATA/data_preparation/01-benchmark_surfaces/${PID}_A.ply" ]; then
  echo "$ID FAIL_01 rc=$rc1 $(grep -oE '[A-Za-z]*Error.*' "$TMPDIR/01.log" | tail -1)"; exit 1
fi
timeout 3600 $SEXEC "$SRC/data_preparation/04-masif_precompute.py" masif_ppi_search "${PID}_A" > "$TMPDIR/04.log" 2>&1
rc4=$?
if [ ! -s "$PRECOMP/${PID}_A/p1_input_feat.npy" ]; then
  echo "$ID FAIL_04 rc=$rc4 $(grep -oE '[A-Za-z]*Error.*' "$TMPDIR/04.log" | tail -1)"; exit 1
fi

REP=$(PYTHONPATH=$ROOT/src $PY -m masif_graph.p6.pl_graph --ids <(echo "$ID") --out "$OUT" \
        --pdbbind "$ROOT/data/pdbbind" 2>&1 | grep '^{' | tail -1)
rm -rf "$PRECOMP/${PID}_A"          # 40 MB/complex; the npz is the only artefact we keep
echo "$ID $REP"
case "$REP" in *'"ok": true'*) exit 0 ;; *) exit 1 ;; esac
