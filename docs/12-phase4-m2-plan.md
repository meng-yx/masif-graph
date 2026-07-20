# Phase 4 — M2 execution plan (the objective gate: holo→AF3 robustness)

> Status: PLAN (awaiting human go for any GPU launch). M0 + M1 are complete (`docs/10` §16): the from-scratch
> heterogeneous GNN, once stabilized (VICReg + frozen τ + T weight-decay), reaches the frozen-MaSIF **holo→holo**
> ceiling (sc per-complex median 0.997; dense beats frozen-dense). That is the *feasibility / do-no-harm* gate —
> **necessary, not sufficient.** M2 is the first Phase-4 milestone that tests the project's **north star**:
> is the representation **robust to sidechain/backbone conformation** (holo DB, AF3 query)?

---

## 0. What M2 decides (design §8 M2, §5.3, §6-Stage-B)

Turn on **conformer-augmented queries** (§5.3): the query-side embedding is drawn from a random conformer
`c ∈ {holo, AF3 sample 1..k}` while the target stays holo, so the InfoNCE + learned-`T` objective must
*recover the true contact regardless of which conformer produced the query*. Invariance falls out of the task
(cannot collapse — hard negatives still separate). Then **evaluate on real AF3** (never only on the training
conformer type — risk §5).

**Two questions M2 answers, both first-class:**
1. **Robustness (the gate).** Does conformer-augmented from-scratch training shrink the holo→AF3 descriptor-
   separation gap relative to (a) frozen MaSIF and (b) the holo-only Stage-A encoder — beating the Phase-3 **M3
   +0.016 bar** — while **preserving holo→holo**?
2. **Does the chemistry graph earn its keep?** Phase-3 M3 found unfreezing helped (+0.016) but the **chem graph
   added nothing**. Phase-4's entire premise (CLAUDE.md north star, D-decisions) is that *connectivity + bond
   rotatability* is what buys conformation-robustness. So the atom-graph ablation is **not optional** — it is a
   primary arm of the matrix. If robustness appears with the graph ablated too, the graph hypothesis is refuted
   a second time and Phase 4's rationale must be re-examined.

---

## 1. The gate table (exact thresholds — decide BEFORE running)

All metrics: descriptor-separation AUC at the surface-atom level, **held-out complexes (complex-level split,
disjoint from all training and from the 31 `m1_ids`)**, on the **induced-fit-only stratum** (structural-mismatch
non-binders excluded — the ~23% domain-swap cases have no query-side fix; design §10). Report **per-AF3-sample
AUC (all 5 diffusion samples) + the spread**, pooled *and* per-complex median. Frozen MaSIF scored on the
**identical** pos/neg pairs as the reference ceiling every eval.

| quantity | Phase-3 reference | **M2 gate** |
|---|---|---|
| **holo→holo learned** (do-no-harm floor) | Stage-A: pooled ~0.90 / sc median ~0.99 | **≥ Stage-A − 0.01** (must not regress) |
| **frozen AF3→holo** (the gap being closed) | pooled ~0.82; addressable induced-fit gap **+0.069** | reference only (re-reproduce on M2 pairs) |
| **M3 unfreeze-beats-frozen bar** | **+0.016** AF3→holo (8/8 seeds +) | must **exceed** this |
| **learned AF3→holo − frozen AF3→holo** (Δ robustness) | — | **PASS ≥ +0.03** pooled (≈2× M3 bar, ~45% of +0.069); **STRONG ≥ +0.05**; majority of seeds positive |
| **per-sample spread** (sd across 5 AF3 samples) | frozen: report | **shrinks vs frozen** (invariance signal) |
| **graph vs no-graph** (Δ from ablation) | M3: ~0 | reported with CI; graph "earns keep" only if **> +0.01 and seed-consistent** |
| shuffled-label control | ~0.50 | **≈0.50** (non-negotiable) |

**Verdict logic.** PASS the M2 gate ⇒ proceed to M3 (scale conformers/data, geodesic ablation, optional pose
scorer). FAIL (Δ ≤ M3 +0.016, or holo→holo regresses, or spread doesn't shrink) ⇒ **stop and diagnose** — the
frozen descriptor's strong ceiling (Phase-2 lesson #3, re-confirmed in M3) would then bound the from-scratch
approach too, and D1-B (the escalation) is on the table.

---

## 2. Data prep (mostly already on disk — this is the cheap part)

Phase-3 already ran the AF3 pipeline (`src/masif_graph/af3/{sequence,relabel,build_pdb,prepare,analyze}.py`;
Jed MSA + Kuma H100 inference) for **~193 complexes**: 31 eval (`logs/phase3/m1_ids.txt`) + 162 train
(`logs/phase3/m3_train150_ids.txt`), each with ~5 diffusion samples, holo-numbered + interface-local Kabsch
superposed. **Start M2 entirely on this existing set — no new AF3 generation to reach the gate.**

Steps:
1. **Build p4 hetero graphs for AF3 conformers.** The p4 dataset (`src/masif_graph/p4/dataset.py`) currently
   loads only `{cid}__holo__p{1,2}.npz`. Run the existing p4 precompute (`p4.precompute`) on the Phase-3 AF3
   surfaces (`logs/phase3/af3_surf_samples/…`, relabelled) to emit `{cid}__af3_s{k}__p{1,2}.npz` beside the
   holo npz. This is the one non-trivial data step; CPU, ~free.
2. **Splits (leak-clean, guardrails-mandatory).** Train / held-out **complex-level, mutually disjoint**, held-out
   disjoint from the 31 `m1_ids`. Reuse the M3 train/eval id lists (already leak-checked by id **and** PDB-stem;
   `docs/07` M3). Verify 0 PDB-stem overlap again before launch (a prior agent caught an `RP/AS/AF` variant leak
   — the holo filter must exclude `RP/AS/AF`, not just `AF`).
3. **Structural-mismatch filter on TRAINING POSITIVES ONLY.** Drop 1A2W-type non-binding-conformation cases with
   the Phase-3 detector (`run_m1_mismatch.py`: retention<0.5 OR interface-local Cα-RMSD>4Å). **Not a deployment
   filter** (needs the holo complex) — training hygiene only (design §6).
4. **Scale-up (only if the gate passes on the ~193 set, deferred to M3):** generate AF3 conformers across more
   of the 4,943 train set — Jed MSA + Kuma H100, reusing Phase-3 `/work/upthomae/Meng/phase3_af3/*.sh`. This is
   the real wall-clock/GPU cost item and stays gated behind a PASS.

---

## 3. Code changes (the real gaps — the pipeline is holo-only today)

| # | file | change | size |
|---|---|---|---|
| C1 | `p4/precompute.py` | emit AF3-conformer npz (`__af3_s{k}__`) from Phase-3 relabelled AF3 surfaces | small |
| C2 | `p4/dataset.py` (`ComplexP4`) | hold conformer variants per chain; **query-side sampler** `c ~ {holo, af3_s0..k}`, target from holo (§5.3); optional **two-conformer positive** (sample two, both must match) | medium |
| C3 | `p4/train.py` | `--conformers <dir>` / stage-B mode (fine-tune from a Stage-A `--init-ckpt`); `--two-conformer`; optional `--aux-consistency <w>` (small `‖z_af3 − z_holo‖²` on identity-matched atoms, §5.3 — ablated, keep ≤0.05) | medium |
| C4 | eval | **AF3→holo AUC for the p4 encoder**, per-AF3-sample + spread, with frozen MaSIF on identical pairs. Adapt `run_m1_af3.py` (currently pooled-frozen / M3 descriptor) to also score a p4 checkpoint's `z` under learned `T`; keep the holo→holo eval as do-no-harm floor | medium |
| C5 | `p4/dataset.py` or a filter script | apply the mismatch filter to training positives (C2 loads only clean positives) | small |

The **stable recipe is fixed** (do not re-tune the optimizer — that fight is won): `--vicreg-var 2.0
--vicreg-cov 0.04 --freeze-tau --tau 0.1 --t-wd 1e-3 --lr 5e-4 --grad-clip 1.0 --cosine --stream --bank 128
--d 64 --d-out 32 --layers 4`. Re-log the diagnostic (z_std, τ, ‖T‖₂, grad-norm) on the first Stage-B run to
confirm collapse stays fixed under conformer augmentation before trusting any AUC.

---

## 4. Training / ablation matrix (cheapest-first, gated)

### Stage B.0 — zero-training robustness probe (run FIRST; ~free)
Take the **existing holo-only Stage-A vicreg checkpoints** — all four present at
`/work/upthomae/Meng/phase4/vicreg_{sc,dense}_best_seed{0,1}.pt` (with matching `vicreg_*_seed*.json` curves) —
and run the new C4 eval: AF3→holo vs frozen, on the held-out set. **Question:** is the from-scratch substrate
*already* more robust than frozen MaSIF, with no invariance training at all? This costs one CPU/short-GPU eval,
reuses the M1 checkpoints, and de-risks everything below — if the substrate is already ≥ frozen it's a strong
prior; if it's *worse*, C2/C3 must carry the whole gate. Report before launching B.1.

### Stage B.1 — core invariance matrix (the gate)
Fine-tune from the Stage-A checkpoint with conformer-augmented queries. **2×2×2 = 8 runs:**

| axis | levels |
|---|---|
| **graph** | full hetero (atom+vertex) vs **no-atom-graph** (vertex-only / covalent edges ablated) — *the chem-graph test* |
| **train-pos** | **sc** (the clean gate metric) — dense deferred to B.2 |
| **two-conformer** | off vs **on** (§5.3 stronger invariance push) |
| **seed** | 0, 1 |

So: {full, no-graph} × {1-conf, 2-conf} × {seed 0,1} at `--train-pos sc`. Each is a short fine-tune (see §5).
Aux-consistency stays **off** in B.1 (add as a B.2 ablation only if the gate is marginal).

### Stage B.2 — only if B.1 PASSES
Add `--train-pos dense`, more seeds (→ report range not best), the aux-consistency ablation, and **scale AF3
conformers** across more of the train set (§2 step 4). This is the budget-gated escalation.

---

## 5. Cost & wall-clock (the ~193-set gate is cheap; scale-up is the commitment)

- **AF3 conformers:** already generated for the 193 complexes ⇒ **CHF 0 new** to reach the gate.
- **AF3→p4 graph precompute (C1):** CPU on Jed, ~free, hours wall-clock.
- **Stage-B fine-tune:** short (fine-tune from Stage-A, ~193 complexes ≪ the 4,811 full set; full-set Stage-A was
  ~1.6 h / 50 ep on 1×H100). Est. **≪ 30 min/run**, 8 runs ⇒ **CHF ~1–2** on Kuma H100.
- **Eval (C4):** CPU/short-GPU, ~free.
- **Total to the M2 gate: ~CHF 2–4**, all inside one session's CHF-100 per-session ceiling (see CLAUDE.md
  Compute — the ceiling is per unattended session, not a project total).
- **B.2 scale-up (deferred, gated):** AF3 generation across the train set = the real GPU + wall-clock item
  (Phase-3: ~5 min/complex MSA on Jed CPU + Kuma H100 inference). Scope only after a PASS.

**Any GPU launch needs explicit human go-ahead** (CLAUDE.md Compute). This plan does not launch anything.

---

## 6. Controls & honesty (ml-research-guardrails — non-negotiable)

- **Shuffled-label ≈ 0.50** every eval; **holo→holo do-no-harm floor** reported every eval; **frozen MaSIF on
  identical pairs** as the ceiling.
- **Complex-level holdout**, no holo/AF3 leakage, held-out disjoint from `m1_ids`; re-verify by **PDB-stem**, not
  just id (prior leak was an `RP/AS/AF` variant).
- **Report the seed range, not the best epoch/seed** (the M1 scale-up's "0.822" was the luckiest spike of a
  diverging run — full per-epoch curves overturned a best-epoch auto-verdict; `docs/10` §15). Read the full
  training curve; a resume conductor must re-verify any auto-written verdict.
- **Per-sample AUC across all 5 AF3 diffusion samples + spread** is the literal sharpened target — not a single
  averaged number.
- Evaluate on **real AF3**, never only on the training conformer type (risk §5).

---

## 7. Immediate next actions (in order)
1. **C1 + C4** — AF3-conformer graph precompute + the AF3→holo p4-encoder eval harness. *(CPU, no launch)*
2. **Stage B.0 zero-training probe** — score the existing `vicreg_*_best_seed*.pt` checkpoints AF3→holo vs
   frozen; report before any training. *(decision checkpoint)*
3. **C2 + C3 + C5** — conformer sampler, Stage-B training mode, mismatch filter.
4. **Stage B.1 8-run matrix** — human go for the (~CHF 2) Kuma launch; diagnostic on run 1.
5. Fill the gate table (§1), write the M2 verdict into `docs/10` / a new results doc; **do not touch M3** until
   the gate is met.
