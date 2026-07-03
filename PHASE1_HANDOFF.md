# Phase-1 Autonomous Agent — Handoff / Mission Brief

You are **Claude (Fable 5, max effort)** running **headless in a 24-hour SLURM batch job**
on a Jed compute node. **No human is available until tomorrow.** You cannot ask questions.
Make reasonable decisions, **document them**, and keep going.

A bash **supervisor** wraps you: whenever you stop before the sentinel file exists, it
resumes you with `claude --continue`. So if you hit a wall, **document it and try
alternatives — do not just exit.** You have many hours; use them.

**Guiding ethos (invoke the `ml-research-guardrails` skill continuously):** *a crash is
cheap, a confident wrong result is expensive.* Adopt an adversarial stance toward your own
good numbers — try to break them before you believe them.

---

## 0. Mission & definition of done

Execute **Phase 1 end-to-end** and produce the deliverable:

**`docs/02-phase1-results.md`** — a results note containing, at minimum:
1. **Per-vertex** descriptor-separation ROC-AUC (the baseline) **and per-atom** ROC-AUC for
   **both `mean` and `max` pooling**, computed on the **same complexes** with the **same**
   positive/negative construction.
2. AUC **stratified by atom exposure** (`n_owned_vertices` bins) + whether a **min-exposure
   filter** recovers any lost AUC.
3. A description of the **positive-vs-negative distance-distribution overlap** (atom vs
   vertex) — a plot saved under `docs/` or `logs/` is ideal, but a quantitative summary is
   the requirement.
4. **M2 global-alignment prototype** results: interface-RMSD vs native + fraction of native
   contacts recovered, on a few complexes. (Only if M1 greenlights — see §6.)
5. An **explicit GO / NO-GO recommendation for Phase 2**, with the reasoning and the
   per-complex AUC spread (not just the pooled number).
6. A short **"Decisions & assumptions"** section listing every judgement call you made.
7. A short **"What was NOT tested / caveats"** section (honest limits).

**When the deliverable is complete AND you have self-verified it (§9 checklist), create the
sentinel so the supervisor stops:**
```bash
touch /scratch/ymeng/masif-graph/logs/PHASE1_DONE
```
If you become **permanently blocked** after genuinely exhausting alternatives, still write
`docs/02-phase1-results.md` documenting what is done, the blocker, and a **provisional**
recommendation — then touch the sentinel.

---

## 1. Read first (authoritative — the design doc IS the spec)

- `CLAUDE.md` (repo root) — project rules, compute layout, the `.sif` reference-tool story.
- `docs/00-context-and-goals.md` — hypothesis, decisions D1–D10, evaluation, risks.
- `docs/01-phase1-design.md` — **your spec**: milestones M0/M1/M2, the per-atom
  representation (§2), pair construction (§4), the go/no-go gate (§3 M1).

Follow the design doc's milestones and definitions. If you diverge, say so explicitly in the
log and the results doc.

---

## 2. Locked decisions (do NOT relitigate)

| Decision | Value |
|---|---|
| Surface descriptor model | **`model_data_paper`** (the published model) — locked by the human |
| N complexes | **40**, drawn from `data/lists/testing.txt`, **deterministic** (fixed seed). Oversample candidates; take the first 40 that preprocess cleanly; **log every skip + reason**. |
| Pooling operators | **mean AND max** (report both) |
| Positive pairs | **vertex-derived contacts** (`01-phase1-design.md §4.2`, <1.0 Å) = primary; **also** report direct heavy-atom <4 Å as a secondary cross-check |
| Scope | **CPU only. No** graph embeddings, **no** descriptor retraining, **no** learned scorer — those are Phase 2+. Phase 1 pools the **frozen** reference net. |

**How to lock `model_data_paper`:** the reference default `model_dir` is
`nn_models/sc05/all_feat/model_data/` which does **not** exist (only `model_data_paper`,
`model_data_1to1`, `model_data_cross_1to1`, …). Create the symlink (reversible), from inside
the reference repo:
```bash
cd /scratch/ymeng/masif-graph/masif-neosurf-af2/masif/data/masif_ppi_search/nn_models/sc05/all_feat
ln -sfn model_data_paper model_data
```
Then **verify** the descriptor step actually loads the paper weights (check logs / the
`model_dir` it prints). Record the confirmation in your progress log.

---

## 3. Milestone 0 — probe inputs via the reference `.sif` (CPU)

The reference pipeline runs **entirely inside** `masif-neosurf-af2/masif-neosurf_v0.1.sif`
(no conda env). `masif-neosurf-af2/` is **its own git repo**, and the reference scripts call
`git rev-parse --show-toplevel` — so **run them from inside `masif-neosurf-af2/`**.

Per complex id `PDBID_C1_C2` (e.g. `1A14_HL_N`) from `testing.txt`:
```bash
cd /scratch/ymeng/masif-graph/masif-neosurf-af2
./masif/data/masif_ppi_search/data_prepare_one.sh   PDBID_C1_C2   # download+triangulate+precompute
./masif/data/masif_ppi_search/compute_descriptors.sh PDBID_C1_C2  # per-vertex desc_straight/_flipped
```
These invoke singularity internally. **Smoke-first: run ONE complex end-to-end and verify
outputs before batching all 40.** Then parallelize across the allocated CPU cores (e.g.
`xargs -P 6`, GNU parallel, or backgrounded jobs) — watch memory (you have ~32 GB).

You need, per chain: the regularized surface `.ply` (**vertices, normals, per-vertex iface
score**) and per-vertex **`desc_straight`** / **`desc_flipped`** (80-D). **Locate the exact
output paths yourself** (look under `masif-neosurf-af2/masif/data/masif_ppi_search/`
— `data_preparation/…`, `descriptors/…`, `output/…`) and record them in the log. Verify
descriptor arrays are `n_vertices × 80`.

> If RCSB download or a specific PDB fails, skip that complex (log it) and draw the next
> candidate — do not let one bad structure stall the batch.

---

## 4. Milestone 1 — the pooling feasibility probe (THE GATE)

Build new code under `src/masif_graph/` (`io/ surface/ pairs/ metrics/ experiments/`) per the
layout in `01-phase1-design.md §6`. Then:

1. **vertex→atom map:** KDTree from surface vertices to the chain's **heavy atoms** (exclude
   H). `a(v)` = nearest heavy atom. **Surface atom** = any atom owning ≥1 vertex. Persist
   `vertex_atom_idx` + a surface-atom table (`atom_id, element, residue, coord,
   n_owned_vertices, owned-vertex list`).
2. **per-atom embeddings:** pool (mean & max) over each atom's owned vertices, **separately**
   for `desc_straight` and `desc_flipped`. **Pool the precomputed `desc_flipped` directly —
   do NOT flip the pooled straight vector** (`§2.2`). Also compute per-atom **normal** =
   unit-normalized mean of owned-vertex normals.
3. **pairs (`§4`):** reproduce reference vertex-contact **positives** (§4.1, nearest
   cross-chain vertex <1.0 Å; optional sc band 0.5–1.0), map to owner atoms → atom positives
   (§4.2, dedup). **Negatives** via `neg_mix` cross/within/hard (the reference uses
   neg_ratio 5, cross 0.5 / within 0.3 / hard 0.2 — reuse or a simple balanced set; document
   your choice). Hard = atoms just outside the contact cutoff.
4. **metric:** descriptor-separation **ROC-AUC**. Score for a candidate pair =
   `|| e_flip(a_A) − e_straight(a_B) ||` (complementarity convention: one partner flipped vs
   the other straight). Positives should have **small** distance, negatives large. Compute
   the **per-vertex baseline** identically (`|| desc_flip(v_A) − desc_straight(v_B) ||`) on
   the **same complexes**.
5. **exposure stratification:** AUC within `n_owned_vertices` bins; test a min-exposure
   filter.

### MANDATORY controls before you believe any AUC (ml-research-guardrails)
- **Shuffled-label control:** shuffle the pos/neg labels → AUC **must collapse to ≈0.5**. If
  it does not, you have a metric/leakage bug — **STOP and fix** before reporting anything.
- **Apples-to-apples:** the per-vertex and per-atom metrics **must** use the **identical**
  complex set and the **identical** pair-construction logic, or the comparison is meaningless.
- **Per-complex spread:** report the distribution of per-complex AUC, not only the pooled
  value (N≈40 is small and noisy).
- Sanity-derive the AUC on a tiny hand-checkable subset once, to confirm the metric code.

### The gate
- **GREENLIGHT** Phase 1 full build if **per-atom AUC ≥ per-vertex AUC − 0.02** (ideally
  ≈ baseline ~0.98 on this metric).
- **ESCALATE / flag NO-GO** if per-atom AUC drops materially (≳0.05); first check whether a
  **min-exposure filter** recovers it. Record the finding either way — a small Phase-1 drop
  is **not** failure (the intended gain is Phase 2).

---

## 5. Milestone 2 — global alignment prototype (only if M1 greenlights; CPU)

Under `src/masif_graph/align/`, on a **few** native complexes, starting from a **randomized
binder pose**:
1. **global correspondences:** over all target×binder surface atoms, candidate pairs with
   surface-embedding distance below a threshold.
2. **robust fit:** Open3D `registration_ransac_based_on_correspondence` (the *correspondence*
   variant) → weighted **Kabsch/Umeyama** on inliers.
3. **refine:** point-to-plane **ICP** using atom coords + per-atom normals.
4. **metric:** interface-RMSD of the recovered pose vs native + fraction of native contacts
   recovered. **No learned scorer** (that is Phase 2).

If M1 is NO-GO, you may **minimize or skip** M2 — document the decision and spend the time
hardening the M1 evidence instead.

---

## 6. Compute, budget & safety rules

> **CORRECTION (2026-07-03, added after the Phase-1 run — read if reusing this as a template).**
> The "CHF 100" below was mis-framed as a *ceiling on this job's footprint*. Its actual meaning:
> a **training/compute budget the agent is meant to SPEND** by launching **child jobs** (full
> training / GPU on Kuma). Phase 1 legitimately needed none (it's a frozen-net CPU probe, no
> training), so this **worker** architecture was fine. **Heavier phases (e.g. Phase 2 retraining)
> should run as a *conductor*:** a small long-lived orchestrator job that submits + monitors
> child training jobs against the CHF 100 budget (right-size, log cumulative cost, stay under).
> See the `slurm-claude-agent` skill → "Two architectures: worker vs conductor".

- **Do the work INSIDE this job.** You are already in a 24 h allocation (multi-core CPU,
  ~32 GB). Run preprocessing here in parallel across cores. **Do NOT `sbatch` new jobs for
  Phase 1** — it is CPU-feasible here and needs no extra jobs.
- **GPU / Kuma is NOT needed for Phase 1** (design decision). Only if you hit a genuine wall
  that requires GPU training (you should not in Phase 1):
  - **Budget: CHF 100 total** for any job submissions. **Log every submission's cost.** Stay
    well under. Never exceed it.
  - Jed→Kuma path (see the `connect-to-kuma` skill): stage data to
    **`/work/upthomae/Meng/JED_TO_KUMA`**, then
    `ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 ymeng@kuma.hpc.epfl.ch`, **rsync** from
    `/work/upthomae/Meng/JED_TO_KUMA` to `/scratch/ymeng/<workdir>` on Kuma, then `sbatch`
    there (`/scratch` is NOT shared between clusters; `/work` and `/home` are).
- **No destructive/irreversible actions.** Don't `git commit` (the project says not to unless
  asked). Only modify `masif-neosurf-af2/` via the `model_data` symlink and its own scratch
  outputs. Everything else, keep to this repo.

---

## 7. Environment setup

- **Reference pipeline needs NO conda env** — it is all inside the `.sif`.
- **Your new Phase-1 code needs only a light scientific stack:** `numpy scipy scikit-learn
  biopython open3d plyfile pandas matplotlib networkx tqdm`. The heavy **`torch` / `torch-
  geometric` / `e3nn`** in `environment.yml` are **Phase 2 — you do NOT need them for the
  gate.** Fastest, least-fragile path: create a **minimal** conda env with just the above
  (use the libmamba solver / `mamba` if the solve is slow), OR run the full
  `conda env create -f environment.yml` if you prefer. **Validate by importing** every
  package you rely on before trusting the env. Then `pip install -e .` to wire up the
  `masif_graph` package. Existing system envs (`MaSIF`, `masif-coherence`) *might* have some
  deps — validate before relying on them.

---

## 8. Progress logging (critical — this is how the human reviews you tomorrow)

Maintain an **append-only** log at **`docs/progress/phase1-log.md`**. Timestamped entries
(absolute times) at every milestone, decision, control-check, and checkpoint — what you did,
what you found, what you decided and why, what's next. **Re-read this log (and this handoff)
whenever the supervisor resumes you**, to reorient after context summarization and avoid
redoing work. Check for repeated/contradictory work each cycle.

---

## 9. Self-verification checklist (run BEFORE touching the sentinel)

- [ ] Shuffled-label control was run and **collapsed to ≈0.5**.
- [ ] Per-vertex and per-atom AUC use the **same complexes** and **same pair logic** (no
      leakage / no apples-to-oranges).
- [ ] No fabricated numbers — every number in the results doc traces to code you ran; the
      exact commands/config are recoverable.
- [ ] Per-complex spread reported; N and any skipped complexes documented.
- [ ] `model_data_paper` was confirmed as the loaded model.
- [ ] The results doc has all 7 required elements from §0 and an explicit GO/NO-GO.
- [ ] "pipeline ran without errors" is stated **separately** from "the result is
      scientifically valid" — you claim only what you earned.

**HALT and write a provisional results doc + sentinel** if: the shuffled control does not
collapse; per-atom AUC implausibly beats per-vertex; you cannot rule out leakage; or you have
exhausted alternatives on a hard blocker. Report uncertainty explicitly.

Good luck. Be rigorous, be honest, and leave a clear trail.
