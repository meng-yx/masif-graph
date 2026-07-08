#!/bin/bash
#SBATCH --job-name=phase3-agent
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/phase3-agent-%j.out

# ============================================================================
# Autonomous Phase-3 CONDUCTOR agent, headless, high-autonomy.
# North star: make MaSIF PPI-search robust on AF3-model queries (not holo).
# Runs CPU work on Jed AND ssh-submits/monitors GPU jobs on Kuma. Staged budget
# CHF 100 / ~48h for arc M0-M2; MUST checkpoint before M3 large spend.
# Brief: PHASE3_HANDOFF.md. Live docs: docs/progress/phase3-log.md + docs/{05-phase3-design,07-phase3-results}.md +
# docs/06-phase3-user-comment.md (async user steering).
# Supervisor loop resumes it across crashes/rate-limits until the sentinel.
# ============================================================================

set +e
export HOME=/home/ymeng
export PATH="/home/ymeng/.local/bin:/home/ymeng/miniconda3/bin:/usr/bin:/bin:$PATH"

REPO=/scratch/ymeng/masif-graph
cd "$REPO" || { echo "[SUP] FATAL: cannot cd to $REPO"; exit 1; }

HANDOFF="$REPO/PHASE3_HANDOFF.md"
SENTINEL="$REPO/logs/PHASE3_ARC1_DONE"
GIVEUP="$REPO/logs/SUPERVISOR_GIVEUP_P3"
# Model: Fable 5 at max effort (user choice; conductor API cost is separate from
# the CHF 100 compute budget). Override via PHASE3_MODEL if ever needed.
MODEL="${PHASE3_MODEL:-claude-fable-5}"
FLAGS="--model $MODEL --dangerously-skip-permissions --effort max --max-turns 600"

START=$(date +%s)
DEADLINE=$(( START + 47*3600 + 30*60 ))   # stop starting iterations 30min before the 48h kill
PER_CALL_TIMEOUT=21600                     # 6h hard cap per claude invocation

rm -f "$SENTINEL" "$GIVEUP"
echo "[SUP] $(date '+%F %T') Phase-3 conductor start on $(hostname); jobid=$SLURM_JOB_ID model=$MODEL"
echo "[SUP] deadline=$(date -d @$DEADLINE '+%F %T')"

# ---- inline preflight -------------------------------------------------------
echo "[PREFLIGHT] internet:"; curl -sS -m 12 -o /dev/null -w "  anthropic=%{http_code}\n" https://api.anthropic.com/v1/messages 2>&1
echo "[PREFLIGHT] claude: $(claude --version 2>&1 | head -1); node: $(node --version 2>&1)"
echo "[PREFLIGHT] singularity: $(singularity --version 2>&1)"
echo "[PREFLIGHT] AF3 sif:   $(ls -la /work/upthomae/Meng/AlphaPulldown/container/alphafold3.sif 2>&1 | awk '{print $5,$NF}')"
echo "[PREFLIGHT] AF3 wts:   $(ls /work/upthomae/Meng/AF3_weights/ 2>&1 | head -1)"
echo "[PREFLIGHT] AF3 dbs:   $(ls -d /work/lpdi/databases/alphafold3_dbs 2>&1)"
echo "[PREFLIGHT] atomsurf env: $(ls -d /work/upthomae/Meng/conda_envs/atomsurf_h100 2>&1)"
echo "[PREFLIGHT] chai env:  $(ls -d /home/ymeng/miniconda3/envs/chai 2>&1)"
echo "[PREFLIGHT] holo list: $(wc -l < $REPO/data/lists/full_list.txt 2>&1) pairs"
echo "[PREFLIGHT] masif sif: $(ls $REPO/masif-neosurf-af2/masif-neosurf_v0.1.sif 2>&1)"
echo "[PREFLIGHT] kuma ssh:"; timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=20 -i ~/.ssh/id_ed25519 ymeng@kuma.hpc.epfl.ch 'echo "  reachable: $(hostname); h100:"; sinfo -s -p h100 2>/dev/null | tail -1' 2>&1 | tail -2
echo "[PREFLIGHT] ---- end ----"

KICKOFF="You are the autonomous Phase-3 CONDUCTOR agent (high autonomy). Read PHASE3_HANDOFF.md in \
FULL, then §2 of docs/04-phase2-results.md (why the graph failed + the metric lesson). Execute the \
mission per the handoff: M0 infra smoke -> M1 measure the real holo->AF3 gap on data/lists (with \
shuffled controls, complex-level holdout, absolute metrics) -> M2 first robustness hypothesis \
(AtomSurf/ensemble/flex-weighted). You DECIDE and proceed without asking permission. Maintain the \
four live docs: docs/progress/phase3-log.md (real-time reasoning, a '## <n>. <title>' header per step, \
written as you go), docs/05-phase3-design.md, docs/07-phase3-results.md, and docs/06-phase3-user-comment.md \
(init near-empty; mirror each work-step header; at EVERY step boundary re-read it and reply under any \
new '### USER:' comment with '### AGENT:'). Follow ml-research-guardrails. Log cumulative CHF and \
every Kuma job-id in progress/phase3-log.md. Budget contract: CHF 100 / ~48h for M0-M2; BEFORE any large \
spend (M3: full-dataset retrain / large-scale AF3 generation) STOP, write a checkpoint + cost ask in \
docs/06-phase3-user-comment.md, and wait. When the M0-M2 arc is complete and self-verified (or you are \
permanently blocked, in which case write the blocker + provisional recommendation), touch $SENTINEL"

RESUME="Resume autonomous Phase-3 development. First re-read docs/progress/phase3-log.md (and \
PHASE3_HANDOFF.md if needed) to recover context, cumulative CHF spend, and any Kuma job-ids you \
already submitted — run 'ssh ... kuma squeue -u ymeng' and reattach; DO NOT double-submit or redo \
finished work. Re-read docs/06-phase3-user-comment.md and respond to any new '### USER:' comment before \
continuing. Keep following ml-research-guardrails (shuffled control ~0.5; complex-level holdout, no \
holo/AF3 leakage; report ABSOLUTE apo/AF3 metrics not just differentials; never fabricate; stay under \
the CHF 100 staged budget and checkpoint before any large M3 spend). Continue toward the M0-M2 \
deliverable in docs/07-phase3-results.md. When complete AND self-verified, or permanently blocked (write \
progress+blocker+provisional recommendation), touch $SENTINEL."

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
[ -f "$REPO/docs/07-phase3-results.md" ] && echo "[SUP] deliverable docs/07-phase3-results.md EXISTS." || echo "[SUP] deliverable MISSING."
