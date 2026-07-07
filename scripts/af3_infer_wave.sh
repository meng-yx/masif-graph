#!/bin/bash
# Submit Kuma AF3 inference for the chains of a complex-list. Writes a wave chains file, submits
# the inference array over it, prints the Kuma job id. Usage: af3_infer_wave.sh <complex_list> <tag>
set -u
clist="$1"; tag="${2:-wave}"
WAVE=/work/upthomae/Meng/phase3_af3/${tag}_chains.txt
# derive unique chains for these complexes
awk -F_ '{print $1"_"$2; print $1"_"$3}' "$clist" | sort -u > "$WAVE"
N=$(wc -l < "$WAVE")
echo "wave '$tag': $(wc -l < "$clist") complexes -> $N chains -> $WAVE"
# submit inference array pointing CHAINS at the wave file
ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 ymeng@kuma.hpc.epfl.ch \
  "sbatch --export=ALL,INF_CHAINS=$WAVE --array=1-${N}%8 /work/upthomae/Meng/phase3_af3/infer_array.sbatch 2>&1 | tail -1"
