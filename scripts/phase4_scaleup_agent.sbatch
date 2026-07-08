#!/bin/bash
#SBATCH --job-name=phase4-scaleup
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase4-scaleup-%j.out

# ============================================================================
# Autonomous Phase-4 CONDUCTOR — SCALE-UP phase (resumes the stopped 65436261).
# Arc-1 is DONE (logs/PHASE4_ARC1_DONE). The user gave GO for the full-scale
# Stage-A run: ~4,700-complex from-scratch GNN, 2x2 (train-pos dense vs sc),
# cosine LR + streaming, on Kuma H100 — does 90 -> ~4,700 close the 0.73->0.947
# gap? Reattaches to running jobs (Jed array 65437518 preprocessing) — must NOT
# double-submit. Live docs: docs/progress/phase4-log.md (RESUME STATE authoritative),
# docs/09-phase4-user-comment.md (async steering), docs/10-phase4-results.md.
# Supervisor loop resumes across crashes/rate-limits until PHASE4_SCALEUP_DONE.
# ============================================================================

set +e
export HOME=/home/ymeng
export PATH="/home/ymeng/.local/bin:/home/ymeng/miniconda3/bin:/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4   # Jed nodes thrash at torch's default 72 threads

REPO=/scratch/ymeng/masif-graph
cd "$REPO" || { echo "[SUP] FATAL: cannot cd to $REPO"; exit 1; }

SENTINEL="$REPO/logs/PHASE4_SCALEUP_DONE"
GIVEUP="$REPO/logs/SUPERVISOR_GIVEUP_P4_SCALEUP"
MODEL="${PHASE4_MODEL:-claude-fable-5}"
FLAGS="--model $MODEL --dangerously-skip-permissions --effort max --max-turns 600"

START=$(date +%s)
DEADLINE=$(( START + 47*3600 + 30*60 ))
PER_CALL_TIMEOUT=21600

rm -f "$SENTINEL" "$GIVEUP"     # NOTE: does NOT touch logs/PHASE4_ARC1_DONE (arc-1 stays complete)
echo "[SUP] $(date '+%F %T') Phase-4 SCALE-UP conductor start on $(hostname); jobid=$SLURM_JOB_ID model=$MODEL"
echo "[SUP] deadline=$(date -d @$DEADLINE '+%F %T')"

echo "[PREFLIGHT] internet:"; curl -sS -m 12 -o /dev/null -w "  anthropic=%{http_code}\n" https://api.anthropic.com/v1/messages 2>&1
echo "[PREFLIGHT] claude: $(claude --version 2>&1 | head -1); node: $(node --version 2>&1)"
echo "[PREFLIGHT] jed queue:"; squeue -u ymeng -h -o "  %.10i %.14j %.8T" 2>&1 | head
echo "[PREFLIGHT] kuma:"; timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=20 -i ~/.ssh/id_ed25519 ymeng@kuma.hpc.epfl.ch 'echo reachable $(hostname); squeue -u ymeng -h | wc -l' 2>&1 | tail -2
echo "[PREFLIGHT] ---- end ----"

KICKOFF="You are the autonomous Phase-4 CONDUCTOR, RESUMED for the SCALE-UP phase (high autonomy, headless). \
Arc-1 is COMPLETE (logs/PHASE4_ARC1_DONE exists; M0 gate PASS; M1 verdict in docs/10-phase4-results.md). The \
interactive user gave GO for the full-scale Stage-A scale-up. \
STEP 0 — RECOVER STATE: read docs/progress/phase4-log.md (the RESUME STATE block is AUTHORITATIVE), then \
docs/09-phase4-user-comment.md (reply under any new '### USER:' comment with '### AGENT:'), then \
docs/10-phase4-results.md. \
STEP 1 — REATTACH, DO NOT DOUBLE-SUBMIT: run 'squeue -u ymeng' (Jed) and \
'ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 ymeng@kuma.hpc.epfl.ch squeue -u ymeng'. The Jed ARRAY job \
65437518 (p4_prep_arr, reference .sif preprocessing of the 4,778 remaining clean training complexes) is \
running — WAIT for it (monitor descriptor counts under masif-neosurf-af2/masif/data/masif_ppi_search/\
descriptors/sc05/all_feat); do NOT resubmit it. \
STEP 2 — then execute the NEXT STEPS in the RESUME STATE: (a) build the full clean training id-list \
(training.txt complexes WITH descriptors; EXCLUDE any AF/RP/AS variant AND any sharing a PDB-stem with the 60 \
held-out stageA_heldout_ids.txt or the 31 logs/phase3/m1_ids.txt) — the leakage guard is mandatory; (b) split \
into <=500 chunks and submit scripts/phase4_precompute_array.sbatch (--array=0-K%500) to build hetero-graph \
npz into /work/upthomae/Meng/phase4/stageA_full_npz; (c) rsync src/ to /work/upthomae/Meng/phase4/src; (d) run \
the Kuma H100 2x2 full-scale Stage-A: --train-pos dense AND --train-pos sc, each with --cosine --stream (needs \
--mem~90G), reporting held-out SC and dense AUC vs the frozen 0.947 ceiling — does 90 -> ~4,700 complexes close \
the gap (data-limited) or plateau (architecture-limited)? \
GUARDRAILS (ml-research-guardrails): shuffled control ~0.5, complex-level holdout, the 31 m1_ids stay UNTOUCHED \
by training, ABSOLUTE metrics + per-complex spread, never fabricate, 'pipeline ran' != 'result valid'. Log \
cumulative CHF and every job-id in docs/progress/phase4-log.md; stay under the CHF 100 session budget. Decide \
and proceed without asking; the user steers async via docs/09. When the full-scale verdict is written in \
docs/10-phase4-results.md AND self-verified (or you are permanently blocked — write blocker + provisional \
recommendation), touch $SENTINEL."

RESUME="Resume the Phase-4 SCALE-UP conductor. Re-read docs/progress/phase4-log.md (RESUME STATE authoritative) \
to recover state, cumulative CHF, and job-ids you already submitted; run 'squeue -u ymeng' and \
'ssh ... kuma squeue -u ymeng' and REATTACH — DO NOT double-submit or redo finished work. Re-read \
docs/09-phase4-user-comment.md and answer any new '### USER:' comment before continuing. Keep following \
ml-research-guardrails (shuffled ~0.5; complex-level holdout; m1_ids untouched; absolute metrics; never \
fabricate; stay under CHF 100). Continue the full-scale 2x2 Stage-A (train-pos dense vs sc, cosine, streaming) \
toward the verdict in docs/10-phase4-results.md. When complete AND self-verified, or permanently blocked, touch \
$SENTINEL."

iter=0
short_streak=0
while true; do
    now=$(date +%s)
    if [ -f "$SENTINEL" ]; then echo "[SUP] $(date '+%F %T') sentinel present -> scale-up complete. Stopping."; break; fi
    if [ "$now" -ge "$DEADLINE" ]; then echo "[SUP] $(date '+%F %T') deadline reached. Stopping."; break; fi
    iter=$((iter+1))
    echo "[SUP] ============================================================"
    echo "[SUP] $(date '+%F %T') === ITERATION $iter (elapsed $(( (now-START)/60 )) min) ==="
    t0=$(date +%s)
    if [ "$iter" -eq 1 ]; then
        timeout "$PER_CALL_TIMEOUT" claude -p "$KICKOFF" $FLAGS; rc=$?
    else
        timeout "$PER_CALL_TIMEOUT" claude --continue -p "$RESUME" $FLAGS; rc=$?
    fi
    t1=$(date +%s); dur=$((t1-t0))
    echo "[SUP] $(date '+%F %T') iteration $iter finished rc=$rc after ${dur}s"
    if [ "$dur" -lt 180 ]; then
        short_streak=$((short_streak+1))
        echo "[SUP] short run #$short_streak (rc=$rc)"
        if [ "$short_streak" -ge 12 ]; then { echo "gave up after $short_streak short runs, rc=$rc $(date '+%F %T')"; } | tee "$GIVEUP"; break; fi
        if   [ "$short_streak" -ge 9 ]; then sleep 1500
        elif [ "$short_streak" -ge 6 ]; then sleep 600
        elif [ "$short_streak" -ge 3 ]; then sleep 120
        else sleep 20; fi
    else
        short_streak=0; sleep 30
    fi
done
echo "[SUP] $(date '+%F %T') scale-up conductor exiting after $iter iteration(s)."
[ -f "$SENTINEL" ] && echo "[SUP] RESULT: sentinel set." || echo "[SUP] RESULT: no sentinel (deadline/giveup)."
