#!/bin/bash
#SBATCH --job-name=phase1-agent
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase1-agent-%j.out

# ============================================================================
# Autonomous Phase-1 development agent (Claude Fable 5, max effort), headless.
# A supervisor loop keeps the agent alive for ~24h: it kicks off the mission,
# then resumes (claude --continue) whenever the agent stops before the sentinel
# file exists, with crash/rate-limit backoff. See PHASE1_HANDOFF.md.
# ============================================================================

set +e  # never let a single failing command kill the supervisor
export HOME=/home/ymeng
export PATH="/home/ymeng/.local/bin:/home/ymeng/miniconda3/bin:/usr/bin:/bin:$PATH"

REPO=/scratch/ymeng/masif-graph
cd "$REPO" || { echo "[SUP] FATAL: cannot cd to $REPO"; exit 1; }

HANDOFF="$REPO/PHASE1_HANDOFF.md"
SENTINEL="$REPO/logs/PHASE1_DONE"
GIVEUP="$REPO/logs/SUPERVISOR_GIVEUP"
MODEL="claude-fable-5"
FLAGS="--model $MODEL --dangerously-skip-permissions --effort max --max-turns 600"

# Wall-clock budget: stop starting new iterations after 23h30m (SLURM kills at 24h);
# leaves the agent time to finish writing the deliverable.
START=$(date +%s)
DEADLINE=$(( START + 23*3600 + 30*60 ))
# Hard cap on a single claude invocation so a hung process can't eat the whole run.
PER_CALL_TIMEOUT=21600   # 6h

rm -f "$SENTINEL" "$GIVEUP"   # clean start
echo "[SUP] $(date '+%F %T') supervisor start on $(hostname); jobid=$SLURM_JOB_ID"
echo "[SUP] repo=$REPO deadline=$(date -d @$DEADLINE '+%F %T')"

KICKOFF="You are the autonomous Phase-1 agent. Read the file PHASE1_HANDOFF.md in full, \
then execute the mission it describes end-to-end. Follow the ml-research-guardrails skill \
throughout. Maintain the append-only log at docs/progress/phase1-log.md. Do not stop until \
the deliverable docs/02-phase1-results.md is complete and self-verified, then create the \
sentinel with: touch $SENTINEL"

RESUME="Resume autonomous Phase-1 development. First re-read docs/progress/phase1-log.md \
(and PHASE1_HANDOFF.md if you need to reorient) to recover context and avoid redoing work. \
Continue toward the deliverable docs/02-phase1-results.md. Keep following ml-research-\
guardrails (shuffled-label control must collapse to ~0.5; per-vertex vs per-atom must be \
apples-to-apples; never fabricate numbers). When the deliverable is complete AND self-\
verified, run: touch $SENTINEL . If you are permanently blocked after exhausting \
alternatives, write docs/02-phase1-results.md documenting progress + blocker + a provisional \
recommendation, then touch $SENTINEL."

iter=0
short_streak=0
while true; do
    now=$(date +%s)
    if [ -f "$SENTINEL" ]; then
        echo "[SUP] $(date '+%F %T') sentinel present -> Phase 1 reported complete. Stopping."
        break
    fi
    if [ "$now" -ge "$DEADLINE" ]; then
        echo "[SUP] $(date '+%F %T') wall-clock deadline reached. Stopping."
        break
    fi

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

    # ---- backoff / give-up logic -------------------------------------------
    # A "short" run (<180s) usually means a transient error or a rate-limit block,
    # not real work. Back off progressively; give up only after many in a row.
    if [ "$dur" -lt 180 ]; then
        short_streak=$((short_streak+1))
        echo "[SUP] short run #$short_streak (rc=$rc)"
        if [ "$short_streak" -ge 12 ]; then
            {
              echo "Supervisor gave up after $short_streak consecutive short runs."
              echo "Last rc=$rc at $(date '+%F %T'). Likely persistent auth/rate-limit/env failure."
              echo "Check the log above and docs/progress/phase1-log.md."
            } | tee "$GIVEUP"
            break
        fi
        # progressive backoff: 3->2min, 6->10min, 9+->25min (rate-limit windows roll over)
        if   [ "$short_streak" -ge 9 ]; then sleep 1500
        elif [ "$short_streak" -ge 6 ]; then sleep 600
        elif [ "$short_streak" -ge 3 ]; then sleep 120
        else sleep 20
        fi
    else
        short_streak=0
        sleep 30   # brief gap between productive sessions
    fi
done

echo "[SUP] $(date '+%F %T') supervisor exiting after $iter iteration(s)."
if [ -f "$SENTINEL" ]; then echo "[SUP] RESULT: sentinel set (agent reported done)."; else echo "[SUP] RESULT: no sentinel (deadline or give-up)."; fi
if [ -f "$REPO/docs/02-phase1-results.md" ]; then echo "[SUP] deliverable docs/02-phase1-results.md EXISTS."; else echo "[SUP] deliverable MISSING."; fi
