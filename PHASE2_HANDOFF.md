# Phase-2 Autonomous Agent — Handoff / Mission Brief (CONDUCTOR)

You are **Claude (Fable 5, max thinking effort)** running **headless in a 24-hour SLURM batch job**
on a Jed compute node. **No human is available until tomorrow.** You cannot ask questions — make
reasonable decisions, **document them**, and keep going.

You are a **CONDUCTOR**, not a lone worker: your Jed job does CPU work *and* **submits + monitors
child GPU training jobs on the Kuma cluster**, spending a **CHF 100 budget**. A bash **supervisor**
resumes you (`claude --continue`) whenever you stop before the sentinel exists — so if you hit a
wall, **document it and try alternatives; do not just exit.**

**Ethos (invoke `ml-research-guardrails` continuously):** *a crash is cheap, a confident wrong
result is expensive.* Try to break your own good numbers before believing them.

---

## 0. Mission & definition of done

Execute **Phase 2 end-to-end** per its spec [`docs/03-phase2-design.md`](docs/03-phase2-design.md)
and produce the deliverable:

**`docs/04-phase2-results.md`** — containing at minimum:
1. **Holo do-no-harm:** the full graph model's descriptor-separation AUC on holo ≥ the Phase-1 atom
   mean-pool baseline (graph must not break holo).
2. **The robustness result (the point):** for each ablation cell, descriptor-separation AUC and
   native-contact recovery on **holo vs apo-like repacked** structures — i.e. how much each model
   **degrades under the repack**. Surface-only is the reference; the graph must degrade **less**.
3. The **ablation** (surface-only / +covalent / +rotatability / +spatial / full) on both states,
   with per-complex spread and ≥3 seeds where feasible.
4. Which **edge features carry the robustness** (attribution), and the **T4 check** (is chemistry
   orthogonal or redundant?).
5. An explicit **GO / NO-GO for Phase 3** and a **D1-B trigger decision** (did the graph improve
   robustness, or must we retrain the surface net atom-centrically?).
6. **Decisions & assumptions** + **What was NOT tested / caveats** + **cumulative CHF spent**.

When complete AND self-verified (§9): `touch /scratch/ymeng/masif-graph/logs/PHASE2_DONE`.
If permanently blocked after exhausting alternatives, still write `docs/04-phase2-results.md` with
what's done + blocker + provisional recommendation, then touch the sentinel.

---

## 1. Read first (authoritative)
- `CLAUDE.md` (north-star: **holo→apo robustness**; compute; the `.sif` reference tool).
- `docs/03-phase2-design.md` — **your spec.** Objective, the single heterogeneous graph, milestones
  M0–M2, the re-targeted gate, the repack methodology.
- `docs/02-phase1-results.md` — what Phase 1 established (frozen mean-pool baseline you build on).
- `docs/00-context-and-goals.md` §5 — decisions D1–D10.

---

## 2. Locked decisions (do NOT relitigate)

| Topic | Locked value |
|---|---|
| **Objective** | **holo→apo robustness**, NOT holo AUC. Holo AUC is a *do-no-harm floor*. A graph showing ~0 gain on holo is expected — the benefit is in the apo-like regime. |
| **Representation** | **One heterogeneous graph** (nodes=atoms; typed edges: **covalent w/ bond order + rotatable/rigid flag**, and **spatial w/ RBF distance**), relational message passing, readout at surface atoms, fused with the **frozen** Phase-1 pooled 80-D descriptor. *Not* two separate A/B embeddings. |
| **Apo-like data** | **Realistic fixed-backbone monomer repack** (PI-confirmed starting point). Split the complex, **repack each chain's sidechains in isolation** (no partner context), **backbone fixed**. Then re-run the reference surface+descriptor pipeline on the repacked chains. Contacts/positives stay defined by the **holo backbone** (unchanged). Tool: **FASPR** (easiest; install it). |
| **D6 / D3 / D2 / D4** | Freeze surface descriptors · D3-A unified contrastive fusion · rotation-invariant graph · reuse Phase-1 sc-filtered contact positives (on holo backbone). |
| **Scope / eval** | Representation-only (pose scorer D7 → Phase 3). Metric = descriptor-separation AUC (holo + repacked) + a small top-k retrieval check. |

**Gate:** do-no-harm on holo **and** full-model degrades under repack **significantly less than
surface-only**, with an attributable rotatability/spatial contribution → **GO Phase 3**. Else
**trigger D1-B**.

---

## 3. STRATEGY DIRECTIVE — CPU-first, guarantee a deliverable, GPU-scale as stretch

**This is the most important instruction. Follow it.**
1. **FLOOR (do this first, guarantee it exists):** get the **entire pipeline working end-to-end on
   CPU at reduced scale** (e.g. 30–50 complexes): env → graphs → repack → surfaces → training →
   the ablation → a **draft `docs/04-phase2-results.md`**. A small, honest CPU result beats an
   unfinished GPU one. Reach this checkpoint early and keep the draft updated.
2. **STRETCH:** once the CPU pipeline is proven and produces a valid ablation, **scale up** (enlarge
   the probe to N≈150–300, more seeds) using **Kuma GPU + the CHF 100 budget**.
3. **Never** let GPU/env/Kuma setbacks prevent the CPU deliverable. If Kuma fights you, ship the CPU
   result and document the GPU plan.

---

## 4. Milestones (recipes)

### M0 — env, tools, graphs, repack, surfaces (CPU)
- **Environment:** the shared env `/work/upthomae/Meng/conda_envs/masif-graph` is **minimal — no
  torch/PyG yet.** Install **PyTorch + PyG (+ torch_cluster/torch_scatter)**. Build a **CUDA** torch
  (works on CPU too) into a **`/work`** env so the same env is reusable on Kuma; validate imports.
  Also install **RDKit, biotite, FASPR**. `pip install -e .`.
- **Atom graphs:** RDKit (primary; in the reference lineage) + biotite template connectivity
  (protein fallback); rotatable-bond flags from RDKit; PyG tensors. Molecule-agnostic; wire proteins.
- **Repack (apo-like):** FASPR fixed-backbone repack of **each chain in isolation** (unbound), holo
  backbone fixed. Regenerate surfaces+descriptors on the repacked chains via the reference `.sif`
  (`cd masif-neosurf-af2`; same tool path as Phase 1; `model_data → model_data_paper` symlink already
  exists). **Smoke ONE complex through the whole holo+repack path before batching.**
- **Enlarge probe** toward N≈150–300 (start smaller for the CPU floor). **Quantify surface-only's
  AUC/contact collapse under repack** — that degradation is the effect the graph must reduce.

### M1 — build graph + fusion; validate pipeline (CPU small-scale)
Implement `graph/` (heterogeneous builder, relational MP, fusion) + `train/` (contrastive loss on
Phase-1 pairs + **rotamer-perturbation augmentation**). **Validate before scaling:** overfit a few
complexes; **shuffled-label control → ~0.5**; **rotation-invariance unit test** (rotate a chain →
fused embedding unchanged).

### M2 — the robustness ablation (the gate)
Train surface-only / +covalent / +rotatability / +spatial / full with rotamer augmentation; evaluate
each on **holo and repacked**. CPU small-scale first (the floor); then scale on **Kuma GPU** (§5).
≥3 seeds where feasible; per-complex spread; shuffled control.

---

## 5. Conductor compute & budget (CHF 100 — spend it, don't exceed it)

- **Your Jed job** does CPU work (M0/M1, small-scale M2). **Submit GPU training to Kuma** for scale.
- **Budget = CHF 100 total** across your Jed job + any Jed child jobs + all Kuma jobs. **Before every
  submission** get its cost: `sbatch --test-only` (Jed) / the printed estimate (Kuma). **Log
  cumulative spend** in the progress log after each. **HALT a submission that would exceed CHF 100.**
- **Kuma path** (see `connect-to-kuma` skill): stage data to **`/work/upthomae/Meng/JED_TO_KUMA`**,
  then `ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 ymeng@kuma.hpc.epfl.ch`, **rsync** it to
  `/scratch/ymeng/<wd>` on Kuma (Kuma `/scratch` is node-local to Kuma; `/work`+`/home` are shared),
  then `sbatch -A upthomae -p h100 -q normal --gres=gpu:1 ...`. GPU env: reuse the CUDA `/work` env
  or build one in a Kuma `build`-QOS job.
- **Reattach, don't double-submit:** record every Kuma job-id in the progress log. On a supervisor
  resume, **check `ssh kuma squeue`** for your recorded ids and pick up results — do not resubmit.
- **No** destructive/irreversible actions; **don't** `git commit`; touch `masif-neosurf-af2/` only
  for its own scratch outputs.

---

## 6. Progress log (critical for coherence + review + budget)
Append-only **`docs/progress/phase2-log.md`**: timestamped entries (absolute times) for every
milestone, decision, control-check, **CHF cost after each submission**, and **every Kuma job-id**.
**Re-read it (and this handoff) on every resume** to reorient and avoid redoing work / double-submits.

---

## 7. Guardrails (ml-research-guardrails — non-negotiable)
- **Leakage:** split by complex (cluster by sequence identity if you can). **A complex's holo and its
  repacked version must be on the SAME side of any split** — never train on holo and test on the
  repacked twin of the same complex. Preprocessing stats fit on train only.
- **Shuffled-label control → ~0.5** on both states; if not, STOP and fix.
- **The robustness metric is *differential*:** report surface-only's holo→repack degradation as the
  reference the graph must beat. "Full model has high holo AUC" is NOT the result; "full model
  degrades less under repack than surface-only, attributably" IS.
- **Never fabricate numbers**; "pipeline ran" ≠ "the graph helps robustness." Report per-complex
  spread and what was not tested. FASPR fixed-backbone repack is a *proxy* for apo — say so.

## 8. HALT (write provisional results + sentinel) if
shuffled control doesn't collapse; you can't rule out leakage; a submission would exceed CHF 100;
or you've exhausted alternatives on a hard blocker.

## 9. Self-verification checklist (before the sentinel)
- [ ] Shuffled control collapsed (~0.5), both states.
- [ ] Split has no holo/repack twin leakage; stats fit on train only.
- [ ] Robustness reported as differential degradation vs surface-only, with per-complex spread.
- [ ] Every number traces to code you ran; commands/config recoverable.
- [ ] Cumulative CHF spend logged and ≤ 100; all Kuma job-ids recorded.
- [ ] `docs/04-phase2-results.md` has all 6 elements (§0) + explicit GO/NO-GO + D1-B decision.
- [ ] "pipeline ran" stated separately from "result is scientifically valid."

Good luck. Ship the CPU floor first, then scale. Be rigorous, honest, and leave a clear trail.
