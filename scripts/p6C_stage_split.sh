#!/bin/bash
# Stage a freshly built split to /work and derive the per-mode Stage-A monitor sets.
# Stage A saves its best checkpoint by held-out AUC, so each mode must be monitored on data it is
# actually trained for — a PL-only model selected on PPI AUC would be selected on noise.
set -eu
SRC="${1:?usage: p6C_stage_split.sh <split_dir>}"
P6=/work/upthomae/Meng/phase6C
mkdir -p "$P6/split"
cp "$SRC"/*.txt "$P6/split/"
head -80 "$P6/split/val_pl.txt" > "$P6/split/val_stageA_plonly.txt"
cp "$P6/split/val_ppi_stageA.txt" "$P6/split/val_stageA_ppionly.txt"
cat "$P6/split/val_ppi_stageA.txt" "$P6/split/val_stageA_plonly.txt" > "$P6/split/val_stageA_combined.txt"
wc -l "$P6"/split/*.txt
