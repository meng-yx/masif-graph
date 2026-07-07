# Phase 3 — async user steering

The agent runs autonomously and does **not** wait for approval per step. You steer here without
interrupting it: under any step header below, add a comment beginning `### 🧑 USER:`. The agent
re-reads this file at **every step boundary** (and periodically during long waits) and replies inline
with `### 🤖 AGENT:` — acknowledging and saying how it will act — then continues. The agent never edits
your lines.

One hard gate: **spend**. The agent stops and writes a checkpoint here (with a cost projection) before
any large M3 spend, and waits for your go.

---

## 1. Context load + orientation

_(agent working; no user comments yet)_

## 2. M0 infra smoke — AF3 CLI pinned + MSA launched

_(no user comments)_

## 3. M0 CLOSED — AF3 end-to-end verified + relabel pipeline works

**Quick status for you:** M0 gate passed on all four tools. AF3 works end-to-end (MSA 11 min on Jed +
inference 68 s on H100; 5-sample ensemble for free). Cost so far ≈ CHF 0.1 of 100. H100 = CHF 0.52/hr,
so budget is not the constraint — wall-clock is. Moving to **M1: the measurement** (holo→AF3 gap) on the
31 Phase-2 complexes. No blocking questions. _(add a `### 🧑 USER:` line here anytime to steer.)_

## 4. M1 generation pipeline built + submitted; eval logic verified

**Status for you:** Whole M1 pipeline is built and de-risked (AF3 relabel → reference surface pipeline
verified to accept AF3 models; eval logic verified to reproduce the Phase-1 holo baseline). AF3 MSA array
(Jed job 65386852, 61 chains) is generating — this is the slow step (~several hours under NFS
contention). Then Kuma inference (fast) → surface batch → the gap number. I'll process complexes in waves
for an early signal. Est CHF so far: ≈0.1 Kuma + ≤5.4 Jed-MSA-worst-case (real ~1-2). Still no blocking
questions; the gap measurement is coming. _(steer anytime with `### 🧑 USER:`.)_

## 5–7. First real AF3→holo gap measured (preliminary, N=5)

**Headline for you:** the pipeline works end-to-end on real AF3 models, and the **first measured
holo→AF3 gap is modest**:
- holo→holo ceiling **0.92**, AF3→holo **0.88** → **gap ≈ +0.04** (descriptor-separation AUC), shuffled
  controls 0.50. Absolute AF3 performance stays well above chance.
- BUT there's a **second gap component** the AUC hides: ~20% of holo interface atoms *stop being surface
  atoms* under AF3's shifted backbone (retention mean 0.80; one homodimer 1A2W lost 100%). Phase-2's FASPR
  couldn't show this (it froze the backbone). So the +0.04 is a **lower bound** on the deployment gap.

This is preliminary (5 complexes). Full ~22-complex run is auto-generating now (final-pass orchestrator;
MSA ETA ~16:55, full result ~17:30). Then I'll stratify by pLDDT/RMSD and start M2 (cheapest lever first:
pLDDT-weighted matching). **No M3-scale spend without checkpointing you first.** Cumulative CHF ≈ 0.2 Kuma
+ ~1-2 Jed MSA. _(steer anytime with `### 🧑 USER:`.)_

### 🧑 USER:
Regarding step 6 "superpose each AF3 monomer onto its holo chain (Kabsch over common CA) before assembling", 
you did the right thing by superimpose each AF3 monomer chain onto the corresponding holo chain. This probably
works for individual cases, but for scaling up you may want to consider superimposing using the interface 
residues only, rather than all residues in each chain, because for large protein chain with large potential
internal flexibility, if you align using all atoms, the interface residues in the moved AF3 model may remain 
misaligned.

### 🤖 AGENT:
Great point, thank you — you're right, and I'll adopt interface-residue superposition. Two clarifications on
where it does/doesn't change the current results, so we're calibrated:

1. **The core M1 gap number is unaffected by the superposition choice.** MaSIF descriptors are computed
   per-chain and are *rotation/translation-invariant* (each chain is triangulated independently; the flip-
   trick + geodesic readout use only local geometry). So how I dock the two AF3 monomers does **not** change
   their descriptors or the descriptor-separation AUC. Whole-chain superposition here only serves to (a) keep
   the two assembled chains from overlapping so `reduce`/MSMS don't choke, and (b) let the reference compute
   `sc_labels` (which I don't use — I map positives from holo). So the +0.03–0.075 gap stands.

2. **Where your point bites — and where I'm fixing it now — is the interface-RMSD *metric* used to stratify
   the gap.** I currently compute interface RMSD by a *whole-chain* Kabsch then measuring the interface
   subset. For a domain-motion case that conflates global motion with local interface change and *inflates*
   the interface RMSD (this is exactly why 1JXQ read 30 Å — a 16.7 Å whole-chain domain error). Superposing
   on the **interface residues only** isolates the true local interface deviation, which is the honest
   stratifier. **I'm implementing interface-local superposition in the RMSD metric now** and will re-report
   the strata with it.

3. **For scale-up (M3):** I'll switch the surface-assembly superposition to interface-based too — not because
   it changes descriptors, but because for large flexible chains a whole-chain fit can leave the moved domain
   overlapping the partner (re-triggering the `reduce` clash) or produce meaningless `sc`. Interface-anchored
   docking is more robust at scale. Noting this in the M3 design.

Acting on (2) now; folding (3) into the M3 plan. Continuing toward the full-set M1.

**Done — and your intuition paid off.** Implemented interface-local superposition in the RMSD metric. On the
current N=7: 1JXQ's interface-RMSD dropped from 30 Å (whole-chain fit) to **2.4 Å (interface-local fit)** —
the 30 Å was a pure domain-motion artifact, exactly as you predicted. And the correlation of the gap with
interface-RMSD **improved from +0.84 → +0.92**, so the local metric is genuinely the better stratifier.
Adopting interface-anchored docking for the M3 scale-up too. Thanks — good catch.

---

## USER NOTE (2026-07-06): AF3 structural-mismatch failure mode — domain swap / context-dependent fold (1A2W)

### 🧑 USER:
Flagging a failure mode you've partly already seen (1A2W losing 100% of its interface surface atoms):
**domain-swapped dimers and other context-dependent folds.** In 1A2W the N-terminal helix (res ~1–22)
swaps to the *partner* monomer in the holo dimer, but AF3 — predicting a lone monomer with no dimer
context — folds it back onto the same chain. The binding-competent conformation is physically **absent**
from the AF3 model, and no superposition (global *or* interface-local) can recover it. This is NOT the
descriptor-robustness problem; it is a structure-prediction limitation, and it is **acceptable for MaSIF
to treat such AF3 models as non-binders.** I don't want these cases silently corrupting either the gap
number or training.

Please handle it explicitly — design the specifics yourself, you have the structures:

1. **Detect with an interface-*local* metric, not a global one.** Global RMSD misses these because the
   folded core aligns fine; only the swapped segment is wrong. You already built interface-local
   superposition — extend it to an interface-patch **conservation** score = local RMSD/lDDT of the
   holo-interface residues *plus* whether they remain surface-exposed. Your "interface atoms stop being
   surface atoms" retention number is essentially the exposure half of this; 1A2W's 100% loss is the
   extreme. **Use 1A2W as a positive control** — if the detector doesn't flag it, the detector is wrong.

2. **Separate addressable from unaddressable.** Stratify the M1 gap by this score and report what
   fraction of the set is structural-mismatch (unaddressable) vs induced-fit (addressable) — that tells
   us how much of the gap a better descriptor could even close.

3. **In M3 training:** drop structural-mismatch cases as *positives* (forcing a match to an absent
   interface is label noise), but keep them as legitimate AF3-side **negatives / expected
   non-retrievals** (the correct answer is "no confident match" — useful for calibration), rather than a
   silent deletion.

4. **Honesty guardrail:** fix the exclusion threshold from *structure* up front (not by what improves the
   score), and always report both filtered and unfiltered numbers.

Not urgent for the current M1/M2 pass — fold it into how you stratify M1 and into the M3 data-cleaning
plan.

### 🤖 AGENT:
This is exactly the right cut, thank you — and it reframes something I'd half-seen. Acting on it now (it's
cheap and it sharpens the M1 verdict). Design + what I found:

**Detector (interface-patch conservation, structure-fixed thresholds, both halves as you said):** for each
complex I classify a chain's holo interface as **structural-mismatch** if EITHER
(a) **exposure loss:** interface-atom retention < 0.5 (holo interface atoms that stop being AF3 surface
atoms — the exposure half; my existing retention metric), OR
(b) **local geometry:** interface-local Cα-RMSD > 4.0 Å (holo-interface residues present but in a
fundamentally different local conformation, beyond induced-fit loop/sidechain motion — the geometry half,
on the interface-local superposition I just built). Thresholds fixed from structure up front, not tuned.
Complex = mismatch if either chain is. **1A2W is the positive control** (retention 0 → must flag).

**Key finding — the separation is already largely happening, and it's reassuring:** the structural-mismatch
cases mostly fall OUT of the intersection-AUC on their own, because losing the interface surface (low
retention) drops them below the ≥8-intersection-positive bar. Of the 5/30 complexes with retention < 0.5
(1A2W 0, 2AOB 0, 4UDM 0.33, 2Z0E 0.38, 3B5U 0.40 — the domain-swap/context-dependent-fold class), only 3B5U
survives into the N=18 usable set. So the **+0.08 intersection-AUC gap is *already* mostly the ADDRESSABLE
(induced-fit) gap**; the unaddressable structural-mismatch cases are captured by the retention metric
(~17% of complexes), not silently corrupting the AUC. I'm making this explicit: reporting (i) the
structural-mismatch fraction as a first-class number, (ii) the gap filtered to induced-fit-only vs
unfiltered, and (iii) 1A2W as a verified positive control.

**M3 data-cleaning (folding in your point 3):** structural-mismatch AF3 models → **dropped as training
positives** (forcing a match to an absent interface is label noise) but **kept as AF3-side negatives /
expected non-retrievals** (correct answer = "no confident match"; good for calibration). Threshold fixed
from structure, both filtered+unfiltered reported. Adding to `05-phase3-design.md §M3`.

Implementing the detector + stratified report now; results below.

**RESULTS (`run_m1_mismatch`, full set of 30):**
- **Structural-mismatch: 7/30 (23%)** — 1A2W, 2AOB, 2IWP, 2PZD, 2Z0E, 3B5U, 4UDM. Five by exposure loss
  (retention <0.5, the domain-swap class incl. 1A2W); **two by local geometry** (interface backbone in a
  very different local conformation despite being exposed: 2IWP 6.5 Å, 2PZD 7.9 Å) — the (b) half earned
  its keep.
- **Positive control 1A2W: FLAGGED (retention 0.00) → PASS.** Detector works.
- **Gap, both reported:** UNFILTERED (N=18) **+0.075**; INDUCED-FIT-ONLY (N=16) **+0.069**. Small
  difference — as I noted, mismatch cases mostly self-exclude via low retention, so only 2 (2PZD, 3B5U)
  were in the usable set. So the **addressable gap a better descriptor could target is ~+0.069**, and
  ~23% of AF3 monomers are unaddressable structural-mismatch (correctly → non-binders).
- Nice side-effect: your metric correctly **clears 1JXQ** (interface-local RMSD 2.4 Å < 4 Å → induced-fit).
  Its 16.7 Å whole-chain motion is *away* from the interface; the interface itself is fine, so its big
  descriptor gap is genuine induced-fit sensitivity, not a fold error. The interface-local view fixed my
  earlier mischaracterisation of 1JXQ.

Folded the M3 data-cleaning (drop-as-positive / keep-as-negative, structure-fixed threshold) into
`05-phase3-design.md §M3`. This completes the M1/M2 arc; writing the final results + M3 checkpoint next.

---

## 🛑 M3 SPEND CHECKPOINT — arc M0–M2 complete; awaiting your go before any M3 spend

**Arc M0–M2 is done and self-verified** (full results in `07-phase3-results.md`; sentinel touched). One-paragraph
summary: the **holo→AF3 descriptor-separation gap is real and moderate** — absolute AF3→holo AUC ≈ 0.82
pooled / 0.85–0.87 median vs a ~0.90 holo ceiling (**gap ≈ +0.08 pooled / +0.06 median**, N=18 of 30, all
controls valid). It splits into an **addressable induced-fit gap (~+0.069)** and **~23% unaddressable
structural-mismatch** (domain swaps → non-binders, per your note). Two **training-free levers** were tested
honestly: pLDDT-weighting (partial, must discard ~50% of atoms) and ensemble soft-min over AF3 diffusion
samples (lossless but modest, gap 0.058→0.038). **Neither closes the gap** → confirms the frozen descriptor's
limited headroom (Phase-2 lesson). The indicated real lever is an **unfrozen/learnable encoder (AtomSurf)
fine-tuned for conformation-invariance** — M0 verified its env + H100 path.

**Total spend so far: ≈ CHF 3** (of the CHF-100 arc budget). No GPU training was run.

**The ask — M3 exceeds the CHF-100/48h arc contract, so I've stopped here.** Proposed staged M3 (details +
anti-circularity + your structural-mismatch data-cleaning in `05-phase3-design.md §M3`):
- **Pilot (recommended first):** AF3-generate ~300–500 training complexes (this exact pipeline), fine-tune
  AtomSurf with a contrastive conformation-invariance objective (pull holo & AF3-multiseed descriptors of the
  same interface together), complex-level holdout + untouched PDBBindplus check, gate on this M1 benchmark.
  **Projected cost ≈ CHF 40–50; wall-clock ~1–2 days** (MSA on Jed is the long pole).
- **Full scale (only if pilot passes):** full `training.txt` generation + retrain + full `testing.txt` +
  top-k retrieval. **Projected ≈ CHF 120–200.**

**Please advise (reply here under a `### 🧑 USER:` line):**
1. Go / no-go for the **M3 pilot** (~CHF 40–50)?  2. Any change to the AtomSurf-first direction (vs, say,
first hardening the ensemble lever or trying Chai/Protenix cross-checks)?  3. Prefer I also compute a
top-k **retrieval** metric on the current M1 set first (cheap, no new GPU) to complement the AUC before M3?

I will **not** spend on M3 until you reply. In the meantime the arc deliverable stands complete.

### 🧑 USER:
Go for M3 pilots. You can clone AtomSurf from https://github.com/Vincentx15/atomsurf.git, but
read and understand how atomsurf constructs atom graph and evaluate if it is appropriate for our purpose before
implementing. You could reuse/adopt code/strategy of the graph from atomsurf, but atomsurf has critical
limitation to what we want: the graph edges in atomsurf is purely based on atomic distance, with no molecular
atom connectivity or bond order information. I would expect the key to train the model to understand apo-holo
relationship is to understand the potential local subtle conformational landscape and how it affects the possible
surface feature landscape, and the conformational landscape depends on the molecular structure like bond order, 
element, electronegativity, valence etc., but atomsurf are not capturing all of these information it its atom
graph. Read Atomsurf repo and think hard about whether it is more efficient to write your own graph or re-use
atomsurf graph before start implementing. You make a judgement call on whether to use Atomsurf as an initial test
based on your own analysis.

### 🤖 AGENT:
Read AtomSurf's graph code (`atomsurf/protein/{atom_graph,graphs}.py`). **You're right, and the case is even
stronger than "missing chemistry":**
- AtomSurf edges = `atom_coords_to_edges(pos, cutoff=4.5)` → KDTree `query_pairs(4.5 Å)`: every atom pair
  within 4.5 Å is an edge, attribute = **distance only**. A covalent C–C (1.5 Å) and an incidental 4 Å
  contact are indistinguishable except by distance. Nodes = element one-hot + pdb2pqr charge + radius (+
  residue amino-type/SSE/hphob). **No bond order, connectivity, electronegativity, valence, hybridization.**
- **Extra reason to reject it — a Phase-2 empirical finding:** in Phase-2 the graph's *distance/spatial*
  edges MOVED with the rotamers and **injected pose-sensitivity → made apo worse**; only the *covalent*
  (invariant) edges were robust. AtomSurf uses **only** distance edges — precisely that failure mode. So a
  distance-only graph is not merely poorer; it's **actively wrong for conformational robustness**.

**Judgment call (proceeding):**
1. **Write our own graph** — adopt Phase-2's already-built chemistry graph (`src/masif_graph/graph/build.py`:
   biotite covalent connectivity + **bond-order** one-hot + **rotatability** flag + flex-depth + element/
   aromatic nodes), enriched with element-derived **electronegativity / valence / hybridization**. This is
   the conformation-**invariant** chemistry anchor. **Not** AtomSurf's graph.
2. **Reuse AtomSurf's learnable SURFACE encoder** (DiffusionNet + surface↔graph comm) — that's its real
   value and the *unfreezing* lever M3 needs (Phase-2 used a frozen descriptor = the ceiling). I'll use it as
   the initial test harness: integration smoke on one of our surfaces, then swap in our chemistry graph.
   **Fallback:** if AtomSurf's data-format/deps are heavier than a lean custom encoder, I build a minimal
   learnable surface encoder over the existing MaSIF surface + our graph. Deciding at the smoke.
3. **Why this isn't Phase-2 redux (which was NO-GO):** the surface encoder is now **unfrozen + co-trained
   end-to-end** with a contrastive holo↔AF3 objective (Phase-2 fused a graph with a *frozen* descriptor at
   readout); and I **drop the pose-sensitive distance edges** that hurt Phase-2.

Full analysis in `05-phase3-design.md §M3 ENCODER + GRAPH DECISION`. **Executing now:** (a) kicking off AF3
generation for a batch of *training* complexes (disjoint from the 30-complex eval set — long-pole MSA
starts now), (b) AtomSurf-surface integration smoke on one of our surfaces + wiring our chemistry graph.
Removing the arc-1 sentinel so the conductor keeps running for M3. Will checkpoint again before the *full*-
scale spend; the pilot stays within the ~CHF 40–50 you approved. 
---

## M3 COMPLETE — result (2026-07-07)

### 🤖 AGENT:
M3 pilot + full training (your GO) are done and self-verified. Honest bottom line:

**Unfreezing the surface encoder works — modestly.** A learnable DiffusionNet surface encoder (residual on
the frozen descriptor, contrastive holo↔AF3 invariance, small reg=0.05), trained on 128 complexes and
evaluated on 30 held-out ones, **robustly beats the frozen descriptor on held-out AF3→holo by +0.016**
(8/8 seeds positive, holo→holo preserved; absolute 0.801→0.817). Real and reproducible.

**But two honest caveats, straight up:**
1. **Your chem-graph hypothesis did NOT pan out at this scale.** I built exactly the graph you asked for
   (covalent connectivity + bond order + rotatability + electronegativity/valence, no distance edges) —
   but graph (+0.016, n=8) ≈ no-graph (+0.016, n=3), statistically indistinguishable. **The unfreezing is
   the driver, not the bond chemistry.** The idea is sound in principle (invariant chemistry → conformational
   landscape), but the learnable surface encoder alone captures what's accessible here; the graph adds no
   measurable signal on top. This may change with a much larger/harder dataset or a graph that feeds the
   *message passing* (AtomSurf-style) rather than a late readout — but I won't claim it works when it doesn't.
2. **Scaling barely helped:** 52→128 complexes moved it +0.014→+0.016. The 14→52 jump was escaping
   overfitting, not a scaling law. The frozen descriptor is a strong ceiling — M3 recovers ~20% of the
   +0.08 gap. Comparable to the *training-free* ensemble-soft-min lever (M2, +0.020 gap-to-ceiling), which
   costs nothing to deploy.

**Recommendation:** the learnable encoder is a real but small lever. The higher-ROI moves are upstream —
filter the ~23% structural-mismatch AF3 models (non-binders) and exploit multi-seed/pLDDT confidence — plus
the free ensemble lever. A materially gap-closing descriptor needs a stronger objective/architecture than
tested, not just more data. Full details + all seeds in `07-phase3-results.md` (M3 section). Total spend ≈
CHF 7 of 100. Touching the M3 sentinel now; happy to push any specific direction (e.g. graph-in-message-
passing, or a much larger training set) if you want.
