#!/bin/bash
# Phase-4 scale-up launcher — robust to conductor-session churn.
# Waits for the precompute-npz array (65442563) to finish, builds the leakage-guarded full training
# id-list from the npz that actually exist, stages it to /work, and submits the Kuma H100 2x2
# (train-pos dense/sc x seed 0/1). Idempotent + lock-guarded + double-submit-guarded, so re-running
# (or a concurrent conductor resume calling it again) never double-submits.
# Run detached:  nohup bash scripts/p4_scaleup_launch.sh &> logs/phase4/scaleup_launch.log &
set -u
REPO=/scratch/ymeng/masif-graph
WORK=/work/upthomae/Meng/phase4
FN=$WORK/stageA_full_npz
PY=/work/upthomae/Meng/conda_envs/masif-graph/bin/python
LOCK=$REPO/logs/phase4/scaleup_launch.lock
PRECOMPUTE_JOB=65442563
KUMA="ssh -o BatchMode=yes -i /home/ymeng/.ssh/id_ed25519 ymeng@kuma.hpc.epfl.ch"
export OMP_NUM_THREADS=4

# --- single-instance lock (stale-safe) ---
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[launch] another instance ($(cat $LOCK)) alive — exiting."; exit 0; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
echo "[launch] $(date) pid=$$ start"

# --- 1. wait for precompute array to leave the queue (cap ~90 min) ---
for i in $(seq 1 360); do
  n=$(squeue -j "$PRECOMPUTE_JOB" -h -r -o %T 2>/dev/null | grep -cE "RUNNING|PENDING")
  [ "$n" -eq 0 ] && { echo "[launch] precompute $PRECOMPUTE_JOB done at $(date)"; break; }
  sleep 15
done

# --- 2. build leakage-guarded train-ids from npz that exist ---
echo "[launch] building train-ids from $(ls $FN/*__holo__p1.npz 2>/dev/null | wc -l) precomputed chains"
$PY - <<PY
import os, glob
REPO="$REPO"; FN="$FN"
def rd(p): return [l.strip() for l in open(p) if l.strip()]
train=rd(f"{REPO}/data/lists/training.txt")
held=set(rd(f"{REPO}/logs/phase4/stageA_heldout_ids.txt"))
m1=set(rd(f"{REPO}/logs/phase3/m1_ids.txt"))
evalstem={c.split('_')[0][:4].upper() for c in held|m1}
# npz present = has BOTH chain graphs + contacts
def has_npz(c):
    return (os.path.exists(f"{FN}/{c}__holo__p1.npz") and os.path.exists(f"{FN}/{c}__holo__p2.npz")
            and os.path.exists(f"{FN}/{c}__contacts.npz"))
ids=[]
for c in train:
    if c in held or c in m1: continue                 # exact eval overlap
    if c.split('_')[0][:4].upper() in evalstem: continue  # PDB-stem leakage guard
    if has_npz(c): ids.append(c)
# assert clean
assert not (set(ids)&held) and not (set(ids)&m1), "LEAK!"
assert not ({i.split('_')[0][:4].upper() for i in ids} & evalstem), "STEM LEAK!"
# also require held-out npz exist (val must load)
held_ok=[c for c in held if has_npz(c)]
open(f"{REPO}/logs/phase4/stageA_full_train_ids.txt","w").write("\n".join(ids)+"\n")
open(f"{REPO}/logs/phase4/stageA_heldout_present.txt","w").write("\n".join(held_ok)+"\n")
print(f"[launch] train-ids={len(ids)} clean (leak-guarded); held-out with npz={len(held_ok)}/{len(held)}")
PY
cp "$REPO/logs/phase4/stageA_full_train_ids.txt" "$WORK/stageA_full_train_ids.txt"
NTR=$(grep -cE '^[A-Za-z0-9]+_' "$WORK/stageA_full_train_ids.txt")
echo "[launch] staged train-ids to /work: $NTR complexes"
if [ "$NTR" -lt 500 ]; then echo "[launch] ABORT: only $NTR train ids — precompute likely incomplete."; exit 1; fi

# --- 3. double-submit guard: any p4_full already on Kuma? ---
RUNNING=$($KUMA 'squeue -u ymeng -h -o "%.16j"' 2>/dev/null | grep -c p4_full)
if [ "${RUNNING:-0}" -gt 0 ]; then echo "[launch] $RUNNING p4_full jobs already on Kuma — NOT resubmitting."; exit 0; fi

# --- 4. submit the 2x2 (train-pos dense/sc x seed 0,1), 60 epochs ---
echo "[launch] submitting Kuma 2x2 at $(date)"
for POS in dense sc; do
  for SEED in 0 1; do
    JID=$($KUMA "sbatch $WORK/phase4_scaleup_stageA.sbatch $POS $SEED 60" 2>&1 | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
    echo "[launch] submitted train_pos=$POS seed=$SEED -> kuma job $JID"
    sleep 2
  done
done
echo "[launch] $(date) 2x2 submitted. Monitor: ssh kuma squeue -u ymeng"
