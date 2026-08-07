#!/bin/bash
#SBATCH --job-name=p6C_agent
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1 --cpus-per-task=8 --mem=32G --time=48:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase6C/agent-%j.out
#
# CONDUCTOR supervisor loop for Phase-6 Workstream C.
# Runs `claude` headless, resumes it on early stops / rate-limits / crashes, and lets the
# agent fan out CHILD jobs (Jed CPU preprocessing + Kuma GPU training). Stops when the agent
# touches logs/PHASE6C_DONE (or logs/PHASE6C_GIVEUP), or ~1h before --time.
set +e
export HOME=/home/ymeng
export PATH=/home/ymeng/.local/bin:/home/ymeng/miniconda3/bin:/usr/bin:/bin:$PATH
cd /scratch/ymeng/masif-graph || exit 2

MODEL=claude-opus-5
FLAGS="--model $MODEL --dangerously-skip-permissions --effort max --max-turns 800"
SENTINEL=logs/PHASE6C_DONE
GIVEUP=logs/PHASE6C_GIVEUP
STARTED=logs/phase6C/_agent_started
mkdir -p logs/phase6C
DEADLINE=$(( $(date +%s) + 47*3600 ))   # stop ~1h before the 48h wall

MISSION="You are the Phase-6 Workstream C autonomous agent. FIRST read PHASE6C_HANDOFF.md in \
full, then docs/progress/phase6C-log.md (its RESUME STATE) and docs/18-phase6C-user-comment.md. \
Then execute Workstream C to completion (definition of done in the handoff §1) as a CONDUCTOR \
with a CHF 100 compute budget: fan out Jed CPU child jobs and Kuma GPU training children, track \
spend in the log, right-size everything. Invoke ml-research-guardrails throughout. Commit+push \
after each working step (/scratch wipe hazard). Do not stop until logs/PHASE6C_DONE exists or \
you are truly blocked; a negative result honestly verified is a valid finish. When done and \
self-verified, touch logs/PHASE6C_DONE."

RESUME="Resume Phase-6 Workstream C. Re-read docs/progress/phase6C-log.md RESUME STATE and \
docs/18-phase6C-user-comment.md (answer any new ### USER comment inline, then keep going). \
Reattach to any child job-ids already recorded (squeue / ssh kuma squeue) — do NOT double-submit. \
Continue to the next step per PHASE6C_HANDOFF.md. Do not stop until logs/PHASE6C_DONE exists."

[ -f "$SENTINEL" ] && { echo "sentinel already present; nothing to do"; exit 0; }

echo "=== supervisor start $(date) host=$(hostname) model=$MODEL ==="
short_streak=0
iter=0
while true; do
  now=$(date +%s)
  [ -f "$SENTINEL" ] && { echo "SENTINEL present -> done $(date)"; break; }
  [ -f "$GIVEUP" ]   && { echo "GIVEUP present -> stopping $(date)"; break; }
  if [ "$now" -ge "$DEADLINE" ]; then
    echo "DEADLINE reached -> stopping (requeue this sbatch to continue) $(date)"; break
  fi
  iter=$((iter+1))
  echo "--- iter $iter start $(date) ---"
  start=$(date +%s)
  if [ ! -f "$STARTED" ]; then
    touch "$STARTED"
    claude -p "$MISSION" $FLAGS
  else
    claude --continue -p "$RESUME" $FLAGS
  fi
  rc=$?
  dur=$(( $(date +%s) - start ))
  echo "--- iter $iter end rc=$rc dur=${dur}s $(date) ---"
  [ -f "$SENTINEL" ] && { echo "SENTINEL present after iter -> done"; break; }
  if [ "$dur" -lt 180 ]; then
    short_streak=$((short_streak+1))
    if [ "$short_streak" -ge 6 ]; then
      echo "6 consecutive short runs -> giving up to avoid hot-loop $(date)"
      touch "$GIVEUP"; break
    fi
    mins=$(( short_streak * 5 )); [ "$mins" -gt 25 ] && mins=25
    echo "short run #$short_streak (likely error/rate-limit); backing off ${mins}min"
    sleep $(( mins * 60 ))
  else
    short_streak=0
    sleep 10
  fi
done
echo "=== supervisor exit $(date) iters=$iter ==="
