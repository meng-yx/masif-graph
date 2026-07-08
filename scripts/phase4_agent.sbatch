#!/bin/bash
#SBATCH --job-name=phase4-agent
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase4-agent-%j.out

# ============================================================================
# Autonomous Phase-4 CONDUCTOR agent, headless, high-autonomy.
# North star: from-scratch heterogeneous GNN (atom+vertex nodes) for
# conformation-invariant interface matching; close the induced-fit gap so any
# of the 5 AF3 samples queries like holo. Unfreezes the whole representation.
# Runs CPU work on Jed AND ssh-submits/monitors GPU jobs on Kuma. Session budget
# CHF 100; MUST checkpoint before the full ~4,943-complex Stage-A run.
# Brief: PHASE4_HANDOFF.md. Plan: docs/08-phase4-design.md. Live docs:
# docs/progress/phase4-log.md + docs/10-phase4-results.md + docs/09-phase4-user-comment.md.
# Supervisor loop resumes it across crashes/rate-limits until the sentinel.
# ============================================================================

set +e
export HOME=/home/ymeng
export PATH="/home/ymeng/.local/bin:/home/ymeng/miniconda3/bin:/usr/bin:/bin:$PATH"

REPO=/scratch/ymeng/masif-graph
cd "$REPO" || { echo "[SUP] FATAL: cannot cd to $REPO"; exit 1; }

HANDOFF="$REPO/PHASE4_HANDOFF.md"
SENTINEL="$REPO/logs/PHASE4_ARC1_DONE"
GIVEUP="$REPO/logs/SUPERVISOR_GIVEUP_P4"
# Model: Fable 5 at max effort (user choice; conductor API cost is via the Max
# subscription, separate from the CHF 100 compute budget). Override via PHASE4_MODEL.
MODEL="${PHASE4_MODEL:-claude-fable-5}"
FLAGS="--model $MODEL --dangerously-skip-permissions --effort max --max-turns 600"

START=$(date +%s)
DEADLINE=$(( START + 47*3600 + 30*60 ))   # stop starting iterations 30min before the 48h kill
PER_CALL_TIMEOUT=21600                     # 6h hard cap per claude invocation

rm -f "$SENTINEL" "$GIVEUP"
echo "[SUP] $(date '+%F %T') Phase-4 conductor start on $(hostname); jobid=$SLURM_JOB_ID model=$MODEL"
echo "[SUP] deadline=$(date -d @$DEADLINE '+%F %T')"

# ---- inline preflight -------------------------------------------------------
echo "[PREFLIGHT] internet:"; curl -sS -m 12 -o /dev/null -w "  anthropic=%{http_code}\n" https://api.anthropic.com/v1/messages 2>&1
echo "[PREFLIGHT] claude: $(claude --version 2>&1 | head -1); node: $(node --version 2>&1)"
echo "[PREFLIGHT] singularity: $(singularity --version 2>&1)"
echo "[PREFLIGHT] masif sif:  $(ls $REPO/masif-neosurf-af2/masif-neosurf_v0.1.sif 2>&1)"
echo "[PREFLIGHT] atomsurf env: $(ls -d /work/upthomae/Meng/conda_envs/atomsurf_h100 2>&1)"
echo "[PREFLIGHT] graph builder: $(ls -d $REPO/src/masif_graph/graph 2>&1)"
echo "[PREFLIGHT] af3 machinery: $(ls -d $REPO/src/masif_graph/af3 2>&1)"
echo "[PREFLIGHT] train list:  $(wc -l < $REPO/data/lists/training.txt 2>&1) complexes"
echo "[PREFLIGHT] m1 eval set: $(ls $REPO/logs/phase3/m1_ids.txt 2>&1)"
echo "[PREFLIGHT] AF3 sif:     $(ls -la /work/upthomae/Meng/AlphaPulldown/container/alphafold3.sif 2>&1 | awk '{print $5,$NF}')"
echo "[PREFLIGHT] kuma ssh:"; timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=20 -i ~/.ssh/id_ed25519 ymeng@kuma.hpc.epfl.ch 'echo "  reachable: $(hostname); h100:"; sinfo -s -p h100 2>/dev/null | tail -1' 2>&1 | tail -2
echo "[PREFLIGHT] ---- end ----"

KICKOFF="You are the autonomous Phase-4 CONDUCTOR agent (high autonomy). Read PHASE4_HANDOFF.md in FULL, \
then docs/08-phase4-design.md in FULL (the complete technical plan), then the lessons in \
docs/07-phase3-results.md (why the frozen-descriptor line is exhausted; the +0.069 induced-fit target; the \
~23% structural-mismatch cases are OUT of scope). Execute arc-1 per the design doc section 8: M0 build the \
heterogeneous graph (atom + surface-vertex nodes; atom-atom covalent, vertex-vertex mesh, vertex-atom edges; \
ALL geometry as SE(3)-invariant scalars) and PASS the rotation-invariance sanity test; then M1 wire the \
Stage-A correspondence trainer (learned symmetric bilinear T + InfoNCE + hard negatives, holo-only), get it \
TRAINING on a small holo subset, measure held-out holo->holo AUC, and estimate full-set Stage-A GPU cost. \
You DECIDE and proceed without asking permission. Maintain the live docs: docs/progress/phase4-log.md \
(real-time reasoning, a '## <n>. <title>' header per step, written as you go), evolve docs/08-phase4-design.md, \
write results to docs/10-phase4-results.md, and docs/09-phase4-user-comment.md (init near-empty; mirror each \
work-step header; at EVERY step boundary re-read it and reply under any new '### USER:' comment with \
'### AGENT:'). Follow ml-research-guardrails: rotation test is a hard gate, absolute metrics always, \
shuffled control ~0.5, complex-level holdout, keep the Phase-3 M1 eval set (logs/phase3/m1_ids.txt) DISJOINT \
from training, never fabricate. Log cumulative CHF and every Kuma job-id in progress/phase4-log.md. Budget \
contract: CHF 100 for this session; BEFORE the full ~4943-complex Stage-A run or large-scale conformer \
generation, STOP, write a checkpoint + cost projection in docs/09-phase4-user-comment.md, and wait. When \
arc-1 is complete and self-verified (M0 done + rotation test passes + M1 pipeline validated with a \
preliminary holo AUC + Stage-A cost estimate + feasibility verdict), or you are permanently blocked (write \
the blocker + provisional recommendation), touch $SENTINEL"

RESUME="Resume autonomous Phase-4 development. First re-read docs/progress/phase4-log.md (and \
PHASE4_HANDOFF.md / docs/08-phase4-design.md if needed) to recover context, cumulative CHF spend, and any \
Kuma job-ids you already submitted -- run 'ssh ... kuma squeue -u ymeng' and reattach; DO NOT double-submit \
or redo finished work. Re-read docs/09-phase4-user-comment.md and respond to any new '### USER:' comment \
before continuing. Keep following ml-research-guardrails (rotation-invariance gate; shuffled control ~0.5; \
complex-level holdout, Phase-3 M1 eval set untouched by training; ABSOLUTE metrics; guard embedding \
collapse; never fabricate; stay under the CHF 100 session budget and checkpoint before the full Stage-A \
run). Continue toward the arc-1 deliverable in docs/10-phase4-results.md. When complete AND self-verified, \
or permanently blocked (write progress + blocker + provisional recommendation), touch $SENTINEL."

iter=0
short_streak=0
while true; do
    now=$(date +%s)
    if [ -f "$SENTINEL" ]; then echo "[SUP] $(date '+%F %T') sentinel present -> arc complete. Stopping."; break; fi
    if [ "$now" -ge "$DEADLINE" ]; then echo "[SUP] $(date '+%F %T') deadline reached. Stopping."; break; fi

    iter=$((iter+1))
    echo "[SUP] ============================================================"
    echo "[SUP] $(date '+%F %T') === ITERATION $iter (elapsed $(( (now-START)/60 )) min) ==="
    t0=$(date +%s)

    if [ "$iter" -eq 1 ]; then
        timeout "$PER_CALL_TIMEOUT" claude -p "$KICKOFF" $FLAGS
        rc=$?
    else
        timeout "$PER_CALL_TIMEOUT" claude --continue -p "$RESUME" $FLAGS
        rc=$?
    fi

    t1=$(date +%s); dur=$((t1-t0))
    echo "[SUP] $(date '+%F %T') iteration $iter finished rc=$rc after ${dur}s"

    if [ "$dur" -lt 180 ]; then
        short_streak=$((short_streak+1))
        echo "[SUP] short run #$short_streak (rc=$rc)"
        if [ "$short_streak" -ge 12 ]; then
            { echo "Supervisor gave up after $short_streak short runs. Last rc=$rc $(date '+%F %T')."; } | tee "$GIVEUP"
            break
        fi
        if   [ "$short_streak" -ge 9 ]; then sleep 1500
        elif [ "$short_streak" -ge 6 ]; then sleep 600
        elif [ "$short_streak" -ge 3 ]; then sleep 120
        else sleep 20
        fi
    else
        short_streak=0
        sleep 30
    fi
done

echo "[SUP] $(date '+%F %T') conductor exiting after $iter iteration(s)."
[ -f "$SENTINEL" ] && echo "[SUP] RESULT: sentinel set." || echo "[SUP] RESULT: no sentinel (deadline/giveup)."
[ -f "$REPO/docs/10-phase4-results.md" ] && echo "[SUP] deliverable docs/10-phase4-results.md EXISTS." || echo "[SUP] deliverable MISSING."
