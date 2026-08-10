#!/bin/bash
# Phase-7 S5 — one AF3 model -> holo-frame PDB -> surface -> AF3 protein npz + contacts.
# Usage: p7_af3_pl_surface.sh <pdbbind_id> <out_npz_dir>
set -u
ID="$1"; OUT="$2"
ROOT=/scratch/ymeng/masif-graph
REFROOT=$ROOT/masif-neosurf-af2; REFDATA=$REFROOT/masif/data/masif_ppi_search
SRC=$REFROOT/masif/source; SIF=$REFROOT/masif-neosurf_v0.1.sif
RAW=$REFDATA/data_preparation/00-raw_pdbs
PRECOMP=$REFDATA/data_preparation/04b-precomputation_12A/precomputation
MODELS=/work/upthomae/Meng/phase7_af3/models
PY=${PY:-/work/upthomae/Meng/conda_envs/masif-graph/bin/python}
CID="pl${ID}"; AFID="${CID}AF"

[ -f "$OUT/${CID}__af3contacts.npz" ] && { echo "$ID DONE(cached)"; exit 0; }
export TMPDIR="${TMPDIR:-/tmp}/p7af_$$_$ID"; mkdir -p "$TMPDIR" "$RAW" "$OUT"
trap 'rm -rf "$TMPDIR"' EXIT

cif=$(find "$MODELS/${CID}_A" -iname "*_model.cif" 2>/dev/null | head -1)
[ -s "$cif" ] || { echo "$ID FAIL no_cif"; exit 1; }
# build_pdb relabels to holo numbering AND superposes into the holo frame (keeps the crystal
# ligand pose valid against the predicted protein)
PYTHONPATH=$ROOT/src $PY -m masif_graph.af3.build_pdb "$CID" A "$cif" "$RAW/${AFID}.pdb" \
  > "$TMPDIR/relabel.log" 2>&1 || { echo "$ID FAIL relabel $(tail -1 $TMPDIR/relabel.log)"; exit 2; }

export PYTHONPATH="$SRC:$REFDATA"
SEXEC="singularity exec --bind $REFROOT:$REFROOT --bind /work:/work --bind $ROOT:$ROOT --bind $TMPDIR:$TMPDIR $SIF python"
cd "$REFDATA" || exit 1
if [ ! -s "data_preparation/01-benchmark_surfaces/${AFID}_A.ply" ]; then
  timeout 3600 $SEXEC "$SRC/data_preparation/01-pdb_extract_and_triangulate.py" "${AFID}_A" \
    > "$TMPDIR/01.log" 2>&1
  [ -s "data_preparation/01-benchmark_surfaces/${AFID}_A.ply" ] || {
    echo "$ID FAIL_01 $(grep -oE '[A-Za-z]*Error.*' "$TMPDIR/01.log" | tail -1)"; exit 3; }
fi
timeout 3600 $SEXEC "$SRC/data_preparation/04-masif_precompute.py" masif_ppi_search "${AFID}_A" \
  > "$TMPDIR/04.log" 2>&1
[ -s "$PRECOMP/${AFID}_A/p1_input_feat.npy" ] || {
  echo "$ID FAIL_04 $(grep -oE '[A-Za-z]*Error.*' "$TMPDIR/04.log" | tail -1)"; exit 4; }

REP=$(PYTHONPATH=$ROOT/src $PY -m masif_graph.p7.pl_af3 --ids <(echo "$ID") --out "$OUT" \
        --pdbbind "$ROOT/data/pdbbind" 2>&1 | grep '^{' | head -1)
rm -rf "$PRECOMP/${AFID}_A"
echo "$ID $REP"
case "$REP" in *'"ok": true'*) exit 0 ;; *) exit 1 ;; esac
