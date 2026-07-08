# Phase 4 — async user-steering channel

> **How this works.** The autonomous agent mirrors each running-log step header here as it works. You (the
> user) may add a comment beginning `### 🧑 USER:` under any header at any time — the agent re-reads this file
> at every step boundary and replies inline with `### 🤖 AGENT:`, then adapts and keeps going. A comment is
> steering, **not** a stop signal. The only routine pause is a **spend checkpoint** (before the full
> ~4,943-complex Stage-A run or large-scale conformer generation), which the agent posts here and waits on.
>
> Leave a comment anywhere below. The agent starts by reading `PHASE4_HANDOFF.md` + `docs/08-phase4-design.md`.

---

<!-- Agent mirrors step headers below this line as work begins. -->

## 1. Orientation — read handoff + design + Phase-3 lessons; mapped reusable assets

Read all four required docs in full. Confirmed the plan: **M0** (hetero graph builder + rotation-invariance
gate) then **M1** (Stage-A correspondence trainer → held-out holo→holo AUC + full-set cost + feasibility
verdict). Mapped reusable Phase-1/2/3 assets (atom-graph builder, surface loaders, contact/negative samplers,
separation-AUC metric, cross-env npz pattern). Established data: 242 holo complexes precomputed, 90-complex
training pool disjoint from the 31-complex Phase-3 AF3 eval holdout. Decided to write the GNN encoder with
torch-core ops only so it runs in both the CPU (Jed) and GPU (Kuma) envs. Now building M0.

## 2. M0 COMPLETE — heterogeneous graph builder + rotation-invariance gate PASS

Built the atom+vertex heterogeneous graph (atom–atom covalent, vertex–vertex mesh, vertex–atom edges; every
geometric feature a distance or cos-angle → SE(3)-invariant). **Rotation gate PASSES**: 180/180 chains built
across the 90-complex pool (0 failures); rotation test 20/20 PASS plus a 12-complex×3-seed run all PASS —
every edge-feature difference under rotation+translation is *exactly 0.0* (invariance is structural). Scale is
tractable (verts/chain median ~4.6k) so no mesh coarsening needed yet. CHF 0. (A parallel M0 gate script
appeared mid-work; I verified it's compatible with my builder and passes, deduped the running log, kept both.)

## 3. M1 — Stage-A correspondence trainer (from-scratch encoder + bilinear T + InfoNCE)

Wiring the Stage-A pipeline: heterogeneous GNN encoder (written), symmetric learned bilinear complementarity
operator T, InfoNCE with hard negatives (holo-only). Plan: CPU smoke (confirm it trains + embedding is
rotation-invariant) → small Kuma H100 run for a held-out holo→holo AUC + full-set GPU cost estimate + the
honest feasibility verdict (can a from-scratch GNN approach MaSIF's ~0.90 holo ceiling?). Will checkpoint here
before any full ~4,943-complex run.

Embedding-level rotation gate PASSED (max|Δ|=0). The CPU smoke trains cleanly (learned AUC beats the frozen
descriptor on identical pairs). A parallel track wrote the training modules on top of my encoder; I verified +
adopted them (they're correct and guardrails-compliant, incl. an honest dual eval: dense contacts + MaSIF's
sc-filtered set where the frozen ceiling is **0.947** — the real ~0.90 gate).

## 4. Stage-A feasibility run on Kuma H100 (held-out holo→holo AUC + GPU cost)

Precomputed 150 complexes to graph-npz, staged to /work, launched the H100 training (90 train / 60 held-out,
3 seeds). First submit hit a NaN from 5 corrupt shape-index values in one training complex (1AKJ); diagnosed
it precisely, sanitized inputs at load + added a NaN-step guard, re-ran. **Kuma job 3795493 running now** — it
will give the held-out holo→holo AUC (learned vs the 0.947 frozen ceiling) + the H100 step time for the
full-set cost estimate. Cumulative spend so far ≈ CHF 0.1. Will post the verdict + a spend checkpoint (for the
full ~4,943-complex scale-up) when it finishes.

## 5. ARC-1 COMPLETE — M0 gate PASS + M1 feasibility verdict + SPEND CHECKPOINT (awaiting go for full-set)

Arc-1 is done and self-verified. **M0**: heterogeneous invariant graph builder + rotation gate PASS
(geometry max-diff exactly 0.0; embedding max|Δz|=0.0). **M1**: the from-scratch GNN (InfoNCE + symmetric
bilinear T) trains on Kuma H100, is provably SE(3)-invariant, and on 60 held-out complexes (3 seeds) reaches
**held-out sc-filtered separation AUC 0.749 ± 0.035 (best epoch)** vs MaSIF's frozen ceiling **0.947** — and
it **beats** the mean-pooled frozen descriptor on dense contacts (0.739 vs 0.682). Controls clean (shuffled
≈0.50). Honest caveat: convergence is **unstable** on only 90 complexes (final-epoch 0.559 ± 0.100; one seed
collapsed) → **promising, data-limited, not yet at MaSIF's ceiling.** Full detail + table:
`docs/10-phase4-results.md`. **Cumulative CHF ≈ 0.4** of 100.

### 💰 SPEND CHECKPOINT — full-set Stage-A run (budget-gated per handoff §1)
Measured cost: **20 ms/complex/step on 1×H100** → full 4,943-complex Stage-A ≈ **99 s/epoch**, so
**100 epochs ≈ CHF 1.4, 150 epochs ≈ CHF 2.1, 3 seeds ≈ CHF 4–6.** Compute is not the constraint. One small
prerequisite before launching: swap `p4.train`'s load-everything-upfront for a **streaming per-complex
loader** (4,943 npz ≈ 14 GB raw won't fit as one in-memory blob). Recommendation: **GO** for the full-set
scale-up — it is the honest, cheap test of whether the ~0.20 gap to 0.947 is data-limited (closes) or
architecture-limited (plateaus), and it gates M2 (invariance). **Holding for your go before launching it**
(this pause is the only routine one). Reply here with GO / adjust / hold.

### 🧑 USER:
GO

### 🤖 AGENT:
Acknowledged — proceeding with the scale-up. One **honest correction** to the checkpoint before I spend: the
"CHF 4–6" costed **training only**, assuming the graph-npz already exist. In reality only **242 holo complexes
have MaSIF reference data** (surfaces + 80-D descriptors + input channels) on disk, of which **151 are usable
for training** (disjoint from the 60 held-out + 31 AF3-benchmark). The literal 4,943-complex set needs the
**MaSIF `.sif` pipeline (MSMS/APBS/descriptor-net) run on ~4,853 more complexes first** — a large CPU
preprocessing job (~minutes/complex → tens of CPU-hours to days), which was NOT in that estimate.

So I'm executing the GO in the **cheapest-first, honest** way:
1. **Fix the dominant caveat first — training instability.** The held-out AUC oscillated (0.45–0.78) under
   constant lr; I've added **cosine-LR decay** so a scale-up result is actually interpretable. A moving,
   unstable model can't tell us if more data helps. Running this on the **same clean 90** (isolates the
   stability fix from data size).
2. **⚠️ Correction (guardrail catch):** I first tried to expand training to "151 available complexes," but on
   inspecting the list they were **RP/AS variants of `1A2W`, `1A99`, … — repack/augmented copies of my EVAL
   complexes**. Training on them = eval leakage. My filter only excluded `AF`, not `RP`/`AS`. **Aborted that.**
   The clean truth: the **90 train.txt complexes are essentially ALL the clean holo training data on disk**;
   there is **no cheap data expansion** — real scale-up genuinely needs the `.sif` preprocessing.
3. **Checkpoint the full reference-preprocessing plan + cost separately** — that (not training) is the real
   "full-set" spend, and it's the only clean path to more data. I'll cost it properly so you can decide. I
   will NOT silently launch thousands of `.sif` jobs.

Running (1) now (cosine-LR stabilization on the clean 90); the preprocessing checkpoint follows. Spend ≈ CHF 0.4.
## 6. Post-GO: cosine fix confirmed + honest full-set cost + gated preprocessing pilot launched

Executing your GO cheapest-first. Three updates:

**1. Instability FIXED (the dominant caveat).** Cosine-LR on the same clean 90/60 (3 seeds): held-out SC
final-epoch AUC **0.559±0.100 → 0.707±0.029** (no seed collapse; final≈best; variance cut ~3–4×). So the
from-scratch GNN gives a **stable, reproducible held-out SC AUC ~0.72–0.74** vs frozen 0.947 — the ~0.2 gap is
now a clean data/architecture question, not training noise.

**2. 💰 Honest cost correction (my §5 estimate was optimistic on two counts):**
- The real "full-set" work is **`.sif` preprocessing of 4,852 complexes** (only 91/4943 are on disk), at
  **~5 min/complex** → ~1 day at 16-way parallel.
- **Jed CPU is billed** (~CHF 0.0055/core-hr) — I'd wrongly assumed free. Full preprocessing ≈ **~CHF 5**,
  + GPU training CHF 2–6 → **full-set ≈ CHF 7–11** (still far under 100, but not free).
- Derisk finding: preprocessing **OOMs on the login node** → it must run as a Jed batch job (done: proper
  sbatch with 128 G).

**3. Gated pilot LAUNCHED (Jed job 65437267): preprocess a first 400 complexes**, then run the **decisive
cheap test — does 90→~450 training complexes lift the stable 0.73?** If yes → data-limited, continue to the
full set (~CHF 5 more). If flat → architecture-limited, and I'll stop and rethink before spending the rest.
This ~CHF 1 test answers the core question before the full ~CHF 7–11 commitment. No user action needed; I'll
report the intermediate-scale result. (Leakage guard kept: the 4,815-id list excludes AF/RP/AS variants and
any complex sharing a PDB-stem with the held-out/eval sets.)

### 🧑 USER:
You should split training set into up to 500 subsets and submit array jobs with a max concurrency of 500. Looping over full dataset on a single thread is too slow

### 🤖 AGENT:
Done — switched to array jobs. Cancelled the single-node loop (job 65437267; the 40 complexes it finished are
kept — preprocessing SKIPs anything already computed). Split the **4,778 remaining** clean training complexes
into **478 subsets** (~10 each) and submitted **`sbatch --array=0-477%500`** (job **65437518**, `standard`
partition, 2 cpu / 12 G / task). With ~478 running concurrently this drops preprocessing from ~40 wall-hr to
**~1 hr** at the same total compute (est ≤ CHF 10.5 reserved, ~CHF 5 actual). Resumable, so any failed/timed-out
task just re-runs the missing ones on resubmit. This preprocesses the **whole** remaining training set (not the
gated 400 pilot) — matching your directive to do it fast. When it lands I'll precompute the hetero-graphs and
run the full-scale Stage-A (cosine LR) → does 90 → ~4,800 complexes close the gap to 0.947? Then M2 invariance.
