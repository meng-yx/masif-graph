#!/bin/bash
#SBATCH --job-name=phase3-m3
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase3-m3-%j.out

# ============================================================================
# RESUME the existing Phase-3 conductor session (moved off the fragile VS Code
# plugin onto a disconnection-proof SLURM supervisor) WITH FULL CONTEXT.
# It picks up its own 1500-message transcript via `claude --resume <SID>` — no
# re-learning. Iteration 1 = --resume <SID>; later iterations = --continue.
#
# PRECONDITION (must be true before submitting): the VS Code conversation with
# this session is STOPPED. A session has a single writer — running here while
# VS Code still drives it will corrupt the transcript.
# ============================================================================

set +e
export HOME=/home/ymeng
export PATH="/home/ymeng/.local/bin:/home/ymeng/miniconda3/bin:/usr/bin:/bin:$PATH"

REPO=/scratch/ymeng/masif-graph
cd "$REPO" || { echo "[SUP] FATAL: cannot cd to $REPO"; exit 1; }

# --- the confirmed AF3 conductor session (VS Code) ---
RESUME_SID="0f6ee918-900c-49d0-a6d8-ce89418ad362"
SESSION_FILE="$HOME/.claude/projects/-scratch-ymeng-masif-graph/${RESUME_SID}.jsonl"

SENTINEL="$REPO/logs/PHASE3_M3_DONE"
GIVEUP="$REPO/logs/SUPERVISOR_GIVEUP_P3M3"
MODEL="${PHASE3_MODEL:-claude-fable-5}"
FLAGS="--model $MODEL --dangerously-skip-permissions --effort max --max-turns 600"
M3_BUDGET="${M3_BUDGET_CHF:-100}"   # CHF ceiling for this M3 arc; pilot ~40-50

START=$(date +%s)
DEADLINE=$(( START + 47*3600 + 30*60 ))
PER_CALL_TIMEOUT=21600

rm -f "$SENTINEL" "$GIVEUP"
echo "[SUP] $(date '+%F %T') Phase-3 M3 RESUME start on $(hostname); jobid=$SLURM_JOB_ID model=$MODEL"
echo "[SUP] resuming session $RESUME_SID"
echo "[PREFLIGHT] session file: $(ls -la "$SESSION_FILE" 2>&1 | awk '{print $5" bytes",$6,$7,$8}')"
[ -r "$SESSION_FILE" ] || { echo "[SUP] FATAL: session transcript not readable — wrong SID or not on /home"; exit 1; }
echo "[PREFLIGHT] internet: $(curl -sS -m 12 -o /dev/null -w '%{http_code}' https://api.anthropic.com/v1/messages 2>&1)"
echo "[PREFLIGHT] kuma: $(timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=20 -i ~/.ssh/id_ed25519 ymeng@kuma.hpc.epfl.ch 'hostname' 2>&1 | tail -1)"
echo "[SUP] M3 budget ceiling: CHF $M3_BUDGET"

KICKOFF="You have been moved off the interactive VS Code plugin onto a headless SLURM supervisor so \
you survive disconnects — your full context is intact (this is a resume of your own session). The \
user APPROVED M3 with a HARD budget of CHF $M3_BUDGET for ALL remaining work combined (AtomSurf \
pilot, full-scale training, generation, everything). Work FULLY autonomously within it — do NOT stop \
to ask for intermediate steps: if the pilot works, go straight on to full training, as long as your \
projected CUMULATIVE spend stays within CHF $M3_BUDGET. ONLY if projected cumulative spend would \
EXCEED CHF $M3_BUDGET must you STOP, post a checkpoint in docs/06-06-phase3-user-comment.md, and wait. \
Track cumulative CHF carefully (you spent ~CHF 3 in M0-M2; Kuma sbatch prints a per-job estimate). \
Continue maintaining your four live \
docs (progress/phase3-log.md real-time, 05-phase3-design.md, 07-phase3-results.md, 06-phase3-user-comment.md) and \
re-read docs/06-06-phase3-user-comment.md at every step boundary, replying under any new '### USER:' with \
'### AGENT:'. Follow ml-research-guardrails (shuffled ~0.5; complex-level holdout; ABSOLUTE AF3 \
metrics; the structural-mismatch/1A2W handling the user asked for; never fabricate). Log cumulative \
CHF and every Kuma job-id. When the pilot is complete AND self-verified, or you are permanently \
blocked (write blocker+recommendation first), touch $SENTINEL"

RESUME_FLW="Resume M3. Re-read docs/progress/phase3-log.md tail + docs/06-06-phase3-user-comment.md for any new \
'### USER:' comment (reply '### AGENT:'), run 'ssh ... kuma squeue -u ymeng' and reattach to any \
running jobs (do NOT double-submit). Continue autonomously within the CHF $M3_BUDGET total budget for \
all remaining work; do NOT stop for intermediate steps — only checkpoint if projected CUMULATIVE \
spend would exceed CHF $M3_BUDGET. Keep guardrails. When M3 is complete and self-verified, or \
blocked, touch $SENTINEL."

iter=0
short_streak=0
while true; do
    now=$(date +%s)
    if [ -f "$SENTINEL" ]; then echo "[SUP] $(date '+%F %T') sentinel present -> M3 pilot done. Stopping."; break; fi
    if [ "$now" -ge "$DEADLINE" ]; then echo "[SUP] $(date '+%F %T') deadline reached. Stopping."; break; fi
    iter=$((iter+1))
    echo "[SUP] ============================================================"
    echo "[SUP] $(date '+%F %T') === ITERATION $iter (elapsed $(( (now-START)/60 )) min) ==="
    t0=$(date +%s)
    if [ "$iter" -eq 1 ]; then
        timeout "$PER_CALL_TIMEOUT" claude --resume "$RESUME_SID" -p "$KICKOFF" $FLAGS
        rc=$?
    else
        timeout "$PER_CALL_TIMEOUT" claude --continue -p "$RESUME_FLW" $FLAGS
        rc=$?
    fi
    t1=$(date +%s); dur=$((t1-t0))
    echo "[SUP] $(date '+%F %T') iteration $iter finished rc=$rc after ${dur}s"
    if [ "$dur" -lt 180 ]; then
        short_streak=$((short_streak+1))
        echo "[SUP] short run #$short_streak (rc=$rc)"
        if [ "$short_streak" -ge 12 ]; then
            echo "Supervisor gave up after $short_streak short runs. Last rc=$rc $(date '+%F %T')." | tee "$GIVEUP"; break
        fi
        if   [ "$short_streak" -ge 9 ]; then sleep 1500
        elif [ "$short_streak" -ge 6 ]; then sleep 600
        elif [ "$short_streak" -ge 3 ]; then sleep 120
        else sleep 20; fi
    else
        short_streak=0; sleep 30
    fi
done
echo "[SUP] $(date '+%F %T') M3 conductor exiting after $iter iteration(s)."
[ -f "$SENTINEL" ] && echo "[SUP] RESULT: sentinel set." || echo "[SUP] RESULT: no sentinel (deadline/giveup)."
