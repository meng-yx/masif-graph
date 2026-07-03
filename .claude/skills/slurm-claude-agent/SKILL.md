---
name: slurm-claude-agent
description: >-
  How to spin up an autonomous, long-running headless Claude Code agent inside a SLURM
  batch job on the Jed cluster, so development continues unattended (e.g. overnight / for
  24h) even when the user's laptop disconnects. Use whenever the user wants to delegate a
  large multi-hour task to a background agent on the cluster, "submit an agent on slurm",
  run Claude headless via sbatch, or keep an agent working while they are away. Covers the
  verified environment facts, the de-risking smoke test, the supervisor-loop pattern that
  survives crashes/rate-limits, the worker-vs-conductor choice (a conductor agent submits &
  monitors child training/GPU jobs against a spend budget), and the handoff-document pattern.
---

# Spinning up an autonomous Claude agent on SLURM (Jed)

Goal: `sbatch` a job that runs `claude` headless for many hours, driving a task to a
concrete deliverable, resuming itself if it stops, and logging everything for later review.
No human is watching — so **de-risk before committing the long job**, and make the agent
**self-supervising**.

## Verified environment facts (Jed, as of 2026-07)
- `claude` CLI: `/home/ymeng/.local/bin/claude` (v2.1.x). `node`:
  `/home/ymeng/miniconda3/bin/node` (v24). Both must be on `PATH` inside the batch job.
- **Auth = Claude Max OAuth** in `~/.claude/.credentials.json` (no `ANTHROPIC_API_KEY`). The
  subscription covers the agent (rate-limited, **not** per-token billed) — so the CHF cluster
  budget is only the **compute** cost. The access token **expires** (~hours) but auto-refreshes
  from the refresh token **as long as the node has internet**.
- **Jed compute nodes HAVE outbound internet** (verified: `api.anthropic.com` reachable,
  RCSB 200). `singularity`/`apptainer` 1.4 present on compute nodes.
- `~/.claude/settings.json` can set defaults, e.g. `{"effortLevel":"max"}`. There is also a
  CLI flag `--effort <low|medium|high|xhigh|max>`.

## The headless invocation
```bash
claude -p "<prompt>" \
  --model claude-fable-5 \          # or another model id
  --dangerously-skip-permissions \  # no TTY to approve tool calls
  --effort max \
  --max-turns 600                   # bounds a single session; resume with --continue
# resume the SAME conversation in the cwd (context preserved):
claude --continue -p "<nudge>" <same flags>
```
`-p` runs the full agentic loop (tools etc.) until the model stops or hits `--max-turns`.
Full tool access is granted by `--dangerously-skip-permissions`. Interactively-authenticated
MCP servers may be absent in headless runs — don't depend on them.

## ALWAYS de-risk first with a 5-minute smoke test
Before a 24h commitment, submit a `--qos=debug` job (2h cap, ~free) that proves, **on a
compute node**: internet to the API + any data source, `singularity` present, and
`claude -p ... --model <m> --max-turns 1` returns a known token with exit 0. If the smoke
test's `claude` prints your sentinel token, the long job's auth/model/network all work.
See `scripts/smoke_test.sbatch` in this repo for a working template.

## The supervisor-loop pattern (survives early stops, crashes, rate-limits)
A single `claude -p` can stop early (task "done", max-turns, transient error, or a Max-plan
rate-limit window). So wrap it in a bash loop that:
1. **Kicks off** iteration 1 with the mission prompt (point it at a handoff file).
2. **Resumes** with `claude --continue -p "<reorient + continue>"` while a **sentinel file**
   does not exist and the wall-clock deadline (set ~30min before `--time`) is not reached.
3. **Backs off** on "short" runs (<~180s = likely error or rate-limit): progressive sleeps
   (2→10→25 min); **gives up** only after many consecutive short runs (avoid hot-looping all
   night). Reset the streak whenever a run does real work.
4. The **agent creates the sentinel** (`touch logs/PHASE1_DONE`) only when the deliverable is
   complete and self-verified — that's the stop signal.

Working template: `scripts/phase1_agent.sbatch` in this repo. Key SLURM bits for Jed:
```
#SBATCH --account=upthomae
#SBATCH --partition=standard
#SBATCH --qos=serial          # single-node, up to 7-day wall (fits 24h multi-core)
#SBATCH --nodes=1 --cpus-per-task=8 --mem=32G --time=24:00:00
#SBATCH --output=/scratch/ymeng/masif-graph/logs/<job>-%j.out   # log to logs/
```
Inside the script: `export HOME=/home/ymeng`, put claude+node+`/usr/bin` on `PATH`, `cd` to
the repo, `set +e` so one bad command can't kill the supervisor.
- **QOS**: this account may access `serial,parallel,bigmem,hugemem,debug` (no `normal`). Use
  `serial` for single-node multi-core; `parallel` only for multi-node.
- **Cost**: 8 cores × 24h on Jed ≈ **CHF 1** (validate with `sbatch --test-only`, which prints
  a cost estimate and the would-be start node without submitting). Always `--test-only` first.

## Two architectures: worker vs conductor (choose per task)
The agent's own SLURM job can either **do the work** or **orchestrate other jobs**. Pick
deliberately — this determines how much compute the agent can actually command.

- **Worker (monolith):** the agent does everything inside its own allocation. Right for
  **lightweight / CPU-bounded** tasks — feasibility probes, analysis, small preprocessing,
  anything that fits one node. Cheap and simple. (Phase 1 here was a worker: a frozen-net
  pooling probe on ~40 complexes — no training, so nothing to fan out.)
- **Conductor:** the agent's job is **small and long-lived** (e.g. 2 cores, low mem — it
  mostly waits) and its purpose is to **submit and monitor CHILD jobs** that hold the real
  compute — full training runs, big multi-node CPU sweeps, or **GPU jobs on Kuma**. Use this
  whenever the task needs **training-scale compute or GPUs** beyond a single node. The heavy
  compute lives in the children, billed separately; the conductor just coordinates.

### Budget semantics (important — don't misread it)
"You have **CHF N** to run jobs" means the agent is **expected to SPEND** it launching child
jobs — it is a **training/compute budget to use**, *not* a ceiling on the agent's own job
footprint. A conductor agent must:
- Before each child job, get its cost estimate (`sbatch --test-only` on Jed; Kuma's `sbatch`
  prints an estimate too) and **track cumulative spend in its progress log**.
- Stay under the cap; prefer the **cheapest sufficient** resource (right-size cores/GPUs/time;
  a debug/short run before a long one); **halt and report** if a run would exceed the budget.

### Submitting child jobs from the conductor
- **Jed child (CPU / big / multi-node):** `sbatch` directly from inside the conductor job
  (nested sbatch works within a cluster). Poll `squeue -j <child>`; react to completion.
- **Kuma child (GPU):** you **cannot** `sbatch` Jed→Kuma. Stage data to shared
  `/work/upthomae/Meng/JED_TO_KUMA`, then over ssh: rsync it to Kuma `/scratch/ymeng/<wd>`
  (Kuma `/scratch` is node-local to Kuma; `/work` + `/home` are shared), then submit there:
  ```bash
  ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 ymeng@kuma.hpc.epfl.ch \
    'sbatch -A upthomae -p h100 -q normal --gres=gpu:1 --time=... /work/upthomae/Meng/<job>.sbatch'
  ```
  See the `connect-to-kuma` skill for partitions/QOS/host-keys. Monitor via
  `ssh ... 'squeue -j <id>'`; back off between polls (don't busy-poll and burn context/tokens).
- The conductor's **supervisor loop still applies** (it must survive its own restarts and
  reattach to already-submitted children by job-id recorded in its log — so it doesn't
  double-submit after a resume).

## The handoff document
The long agent needs a self-contained brief, because it can't ask questions and its context
gets summarized over many hours. Write a `HANDOFF.md` (template: `PHASE1_HANDOFF.md`) with:
- **Autonomy contract**: you're headless, no human until <when>; decide + document + keep
  going; the supervisor will resume you — don't just exit on a wall.
- **Definition of done** + the exact **deliverable path** + the **sentinel** to touch.
- **Locked decisions** (so it doesn't relitigate) and **read-first** pointers.
- **Compute model**: state explicitly whether it's a **worker** or a **conductor**, and if a
  conductor, its **spend budget** (e.g. "CHF 100 to launch training jobs — spend it, don't
  hoard it; log cumulative cost; right-size and stay under") and where children run (Jed / Kuma).
- **Concrete recipes / commands**, **safety rails**, **env setup**, and a mandatory
  **append-only progress log** (`docs/progress/*.md`) it re-reads on every resume.
- A **self-verification checklist** and honest **HALT conditions** (invoke
  `ml-research-guardrails` for any ML work).

## Monitoring / handing back
- `squeue -j <id>`, `tail -f logs/<job>-%j.out`, and the agent's progress log.
- Stop it early: `scancel <id>`. The sentinel file (or a `SUPERVISOR_GIVEUP` file) tells you
  how it ended.

## Gotchas
- Forgetting `HOME`/`PATH` in the batch env → claude can't find node or its config.
- No internet on a compute node would break both auth-refresh and data download — **verify in
  the smoke test**, don't assume.
- Reference tools that use `git rev-parse --show-toplevel` must be run from **inside their own
  git repo**, not a parent repo.
- A huge `-p` prompt is fine, but it's cleaner to keep the prompt short and have the agent
  **read the handoff file** itself (gets it into context via the file tool).
