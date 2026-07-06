#!/bin/bash
#SBATCH --job-name=phase2-agent
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase2-agent-%j.out

# ============================================================================
# Autonomous Phase-2 CONDUCTOR agent (Claude Fable 5, max effort), headless.
# Does CPU work on Jed AND submits/monitors child GPU training jobs on Kuma,
# spending a CHF 100 budget. Supervisor loop resumes it until the sentinel.
# Brief: PHASE2_HANDOFF.md   Spec: docs/03-phase2-design.md
# ============================================================================

set +e
export HOME=/home/ymeng
export PATH="/home/ymeng/.local/bin:/home/ymeng/miniconda3/bin:/usr/bin:/bin:$PATH"

REPO=/scratch/ymeng/masif-graph
cd "$REPO" || { echo "[SUP] FATAL: cannot cd to $REPO"; exit 1; }

HANDOFF="$REPO/PHASE2_HANDOFF.md"
SENTINEL="$REPO/logs/PHASE2_DONE"
GIVEUP="$REPO/logs/SUPERVISOR_GIVEUP_P2"
MODEL="claude-fable-5"
FLAGS="--model $MODEL --dangerously-skip-permissions --effort max --max-turns 600"

START=$(date +%s)
DEADLINE=$(( START + 23*3600 + 30*60 ))   # stop starting iterations 30min before the 24h kill
PER_CALL_TIMEOUT=21600                     # 6h hard cap per claude invocation

rm -f "$SENTINEL" "$GIVEUP"
echo "[SUP] $(date '+%F %T') Phase-2 conductor start on $(hostname); jobid=$SLURM_JOB_ID"
echo "[SUP] deadline=$(date -d @$DEADLINE '+%F %T')"

# ---- inline preflight (log environment health immediately) -----------------
echo "[PREFLIGHT] internet:"; curl -sS -m 12 -o /dev/null -w "  anthropic=%{http_code} " https://api.anthropic.com/v1/messages 2>&1; curl -sS -m 12 -o /dev/null -w "rcsb=%{http_code}\n" https://files.rcsb.org/download/4ZQK.pdb 2>&1
echo "[PREFLIGHT] singularity: $(singularity --version 2>&1)"
echo "[PREFLIGHT] claude: $(claude --version 2>&1 | head -1); node: $(node --version 2>&1)"
echo "[PREFLIGHT] kuma ssh:"; timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=20 -i ~/.ssh/id_ed25519 ymeng@kuma.hpc.epfl.ch 'echo "  reachable: $(hostname)"' 2>&1 | tail -1
echo "[PREFLIGHT] model_data symlink: $(readlink masif-neosurf-af2/masif/data/masif_ppi_search/nn_models/sc05/all_feat/model_data 2>&1)"
echo "[PREFLIGHT] ---- end ----"

KICKOFF="You are the autonomous Phase-2 CONDUCTOR agent. Read PHASE2_HANDOFF.md in full, then \
execute the mission end-to-end following docs/03-phase2-design.md. Follow ml-research-guardrails. \
Maintain docs/progress/phase2-log.md (log every decision, CHF cost, and Kuma job-id). Priority: get \
the full pipeline working end-to-end on CPU at reduced scale and a draft docs/04-phase2-results.md \
FIRST (the floor), then scale on Kuma GPU within the CHF 100 budget. Do not stop until \
docs/04-phase2-results.md is complete and self-verified, then: touch $SENTINEL"

RESUME="Resume autonomous Phase-2 development. First re-read docs/progress/phase2-log.md (and \
PHASE2_HANDOFF.md if needed) to recover context, your cumulative CHF spend, and any Kuma job-ids you \
already submitted — check 'ssh ... kuma squeue' for them and reattach; DO NOT double-submit or redo \
finished work. Continue toward docs/04-phase2-results.md. Keep following ml-research-guardrails \
(shuffled control ~0.5; no holo/repack split leakage; robustness = differential degradation vs \
surface-only; never fabricate; stay under CHF 100). When the deliverable is complete AND \
self-verified, touch $SENTINEL . If permanently blocked, write docs/04-phase2-results.md with \
progress + blocker + provisional recommendation, then touch $SENTINEL."

iter=0
short_streak=0
while true; do
    now=$(date +%s)
    if [ -f "$SENTINEL" ]; then echo "[SUP] $(date '+%F %T') sentinel present -> Phase 2 complete. Stopping."; break; fi
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
[ -f "$REPO/docs/04-phase2-results.md" ] && echo "[SUP] deliverable docs/04-phase2-results.md EXISTS." || echo "[SUP] deliverable MISSING."
