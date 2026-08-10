#!/bin/bash
# Phase-7 — submit the 3-axis gate for every checkpoint, plus the ligand-axis robustness eval.
# Each model is scored on ITS OWN representation: Phase-6C checkpoints on Phase-6C ligand graphs
# (no vertices, no composite arm), Phase-7 checkpoints on Phase-7 ones.
set -u
R=/scratch/ymeng/masif-graph
P6=/work/upthomae/Meng/phase6C
P7=/work/upthomae/Meng/phase7
cd $R
sub() {  # tag ckpt pl_data neosurf_data composite
  [ -s "$2" ] || { echo "SKIP $1 (no ckpt $2)"; return; }
  j=$(sbatch --parsable $R/scripts/p7_gate.sbatch "$1" "$2" "$3" "$4" "$5" 2>&1 | tail -1)
  echo "GATE $1 -> $j"
}
# Phase-6C representation (no ligand vertices -> no composite arm)
sub p6comb_s0  $P6/ret_combined_best.pt   $P6/npz_pl $P6/npz_neosurf ""
sub ppionly_s0 $P6/ret_ppionly_best.pt    $P6/npz_pl $P6/npz_neosurf ""
sub plonly_s0  $P6/ret_plonly_best.pt     $P6/npz_pl $P6/npz_neosurf ""
sub p6comb_s1  $P7/ret_p6comb_s1_best.pt  $P6/npz_pl $P6/npz_neosurf ""
sub ppionly_s1 $P7/ret_ppionly_s1_best.pt $P6/npz_pl $P6/npz_neosurf ""
# Phase-7 representation (ligand surfaces + composite neosurface query)
sub p7comb_s0  $P7/ret_p7comb_s0_best.pt  $P7/npz_pl $P7/npz_neosurf $P7/npz_composite
sub p7comb_s1  $P7/ret_p7comb_s1_best.pt  $P7/npz_pl $P7/npz_neosurf $P7/npz_composite
