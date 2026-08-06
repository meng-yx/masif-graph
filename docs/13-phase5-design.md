# Phase 5 — Deployment-faithful retrieval: does the invariant encoder recover the true binder from AI-predicted structures?

> Living, honest, design-ahead-of-code. Every planned number ties to an artifact + a recoverable
> command. "Pipeline ran" is stated separately from "result is valid." This doc is written before
> the code; it locks the gate and the D-decisions, and is open to a `14-phase5-user-comment.md`
> feedback pass (same convention as Phase 4).

## 0. One-paragraph summary

Phase 4 proved (on small held-out sets) that the from-scratch, SE(3)-invariant encoder is
**conformation-robust** (holo→AF3 top-5 drop ~0.03 vs frozen ~0.14) and, once the DC-offset
normalization bug was fixed (§24 of `10-phase4-results.md`), a **better retriever at scale** than
frozen MaSIF descriptors. Phase 5 asks the **deployment question at full scale on the right
domain**: given a target interface — **experimental *or* AI-predicted (AF3)** — can the encoder
**retrieve the correct binding partner** out of the full MaSIF-search PDB list and its **AF3 apo
counterparts**, more accurately and more robustly than frozen MaSIF? The task's defining property
(§2) is that **every realistic query is effectively apo** with respect to an *unknown* partner, so
conformation-robustness is not an edge case — it is the whole game. Ligands/neosurfaces and the
full TED human-domainome inference are **explicitly deferred to Phase 6**.

## 1. Why Phase 5 now (what Phase 4 left open)

Phase 4 established, but only on **small, partly-leaky** benchmarks:
- The learned encoder is conformation-invariant and reaches the frozen holo ceiling once trained
  full-set with the anti-collapse recipe (`10` §16, §22).
- With DC-offset centering it **matches frozen on small easy DBs and beats it at "thousands"
  scale / dense patches**, where frozen collapses toward random (`10` §23).

Three gaps remain, and Phase 5 closes them:
1. **Scale + leak.** The strongest §23 result used decoys the encoder had *seen in training* — a
   mild optimism asterisk. Phase 5 evaluates on a **held-out test split with sequence-cluster
   disjointness** (§4) and clean decoys — not merely the complex-level 959/4,943 split, which
   leaks homologs (Stage-C draft §6).
2. **The apo side was tiny.** Phase 4's AF3 set was ~31 complexes (`/work/upthomae/Meng/phase3_af3`,
   26 entries). Phase 5 generates AF3 apo models across the **full test set** so the robustness
   claim rests on ~hundreds, not ~31.
3. **The retrieval directions were partial.** Phase 4 tested mainly AF3-query → holo-DB. Deployment
   needs **all four query×DB structure combinations** (§5), especially **AF3-query → AF3-DB**.

**Relationship to the Stage-C draft (`docs/11-phase4-stageC-ppi-scoring.md`).** Stage C is an
*unstarted* 2026-07-07 brainstorm for the **protein-level false-positive / precision** problem — a
three-stage funnel (GNN screen → OT/spectral correspondence scorer → restraint-guided co-folding).
**Phase 5 is that funnel's Stage 1 only** (recall/ranking of the true binder), validated for
conformation-robustness. Stage C's Stages 2–3 (the FP-killing OT scorer + Boltz-2/Chai co-folding
precision) and its **hard-negative** stress test (crystal-contact decoys, swapped-partner, Negatome)
are a **separate later phase** (§10) that consumes Phase 5's validated retriever. The one thing we
pull *forward* from Stage C into Phase 5 is its **sequence-cluster leakage discipline** (§4).

## 2. The deployment task, precisely (record this — it reframes the north star)

At inference (Phase 6): a **database of AI-predicted structures** (the TED human domainome — AF2
models of single folding units, `/work/upthomae/Meng/TED_human_domainome_MaSIF/`, ~25.7k
intracellular / ~39.6k total) is queried by **one target interface** (AF2/AF3/chai or experimental)
to find domains that bind it. The frozen-MaSIF descriptors for that DB **already exist**
(`.../descriptors/sc05/all_feat`); what does not exist is the learned encoder run over it. **That
DB is inference-only and out of Phase-5 scope** — its domain boundaries don't match the PDB list,
and not all list partners are human, so it lacks reliable known-positive pairs.

**The load-bearing insight (user, this session).** A known holo of *A·B* is **not** optimized for a
*different, unknown* binder *C*. If you already had *A·C* you would not be searching. So for the
purpose of *discovering* a new binder, a holo structure (bound to some other partner) is **as
"apo" as an AF2 monomer** — neither is in the conformation optimized for the partner you seek.
⇒ **Conformation-robustness is intrinsic to the task, not a robustness "nice-to-have."** This is
exactly the property Phase 4 measured, so **the north star and the deployment task coincide.**

## 3. The Phase-5 gate (sharpened; supersedes the Phase-4 gate)

> **Gate.** On the **sequence-cluster-clean** held-out test set (§4; the cluster-pruned subset of the
> 959, + AF3 apo counterparts + clean decoy pool), the
> learned encoder retrieves the **true partner** with accuracy **≥ frozen MaSIF** on the do-no-harm
> holo→holo cell, **and** with a **smaller holo→AF3 degradation** than frozen in the AF3-involving
> cells (§5) — i.e. it is **at least as good and more robust** on the deployment-realistic
> directions. The headline cell is **AF3-query → AF3-DB** (fully predicted, the real use case).

Holo→holo is a **floor, not the objective** (as in every prior phase). A learned encoder that ties
frozen on holo→holo but degrades far less on AF3→AF3 **passes**. A learned encoder that wins
holo→holo but degrades as much as frozen under AF3 **fails** — it hasn't earned deployment.

## 4. Data strategy

**Holo (experimental) — reuse verbatim.** The MaSIF-search list, already in the reference repo:
`masif-neosurf-af2/masif/data/masif_ppi_search/lists/{training,testing}.txt` = **4,943 / 959**
(0 overlap, verified). IDs are `PDBID_sideA_sideB` (sides may be multi-chain, e.g. `1A14_HL_N`).
Surfaces + frozen 80-D descriptors are produced by the reference `.sif` stack (per D10 — regenerate,
don't reuse legacy artifacts).

**Apo (AI-predicted) — generate ourselves.** For each **side** of each complex, an AF3 prediction of
that side **in isolation** (unbound-like conformation) = its apo model. Infra already exists and is
proven at small scale: `scripts/af3_msa_array.sbatch` → `scripts/af3_infer_wave.sh`, local weights
`/work/upthomae/Meng/AF3_weights/af3.bin.zst`. Scaling from ~31 to ~959 test complexes (~2×959
sides, minus redundancy) is the **dominant Phase-5 cost** (§8) and is **gated**.

**Split & leak discipline — sequence-cluster holdout (upgraded from complex-level, per Stage-C §6).**
The canonical `training.txt`/`testing.txt` split is **complex-level only**, which *leaks homologs*:
a test partner can have a ≥30%-identity twin in training, so the encoder is not truly naive to it.
Phase 5 therefore enforces **sequence-cluster disjointness**: cluster all chains across the 5,902
complexes at **~30% identity** (mmseqs2), assign each complex to its members' clusters, and keep in
the **eval set only complexes whose every chain's cluster is absent from training**. This will
*prune* the raw 959 down to a cluster-clean subset (report the surviving count — it is the honest
denominator). AF3 test models never enter training; DC-offset centering is mandatory on the learned
encoder (without it it collapses to chance, §24). Controls every eval: shuffled-label (≈ chance),
holo→holo frozen ceiling on **identical** pos/neg pairs, `z_std` sanity assert. **Building the
cluster-clean split is an M0 deliverable** (§7) and gates every downstream number.

## 5. Evaluation — the query×DB retrieval matrix

Retrieval = a **query side's interface patch** ranks all candidate sides in the DB pool by
complementarity score; the **true partner** should rank at the top. Positives = the real partner
side; negatives = all other sides + a held-out decoy pool. Report **top-1, top-5, top-10, MRR,
median rank, and per-complex spread** (never a single top-k point — Phase 4 §21 shows those swing).

Four cells, from the **query structure** × **DB structure** each being holo or AF3:

| cell | query | DB | role |
|---|---|---|---|
| **HH** | holo | holo | do-no-harm floor; reproduces classic MaSIF-search (harness sanity) |
| **AH** | AF3 | holo | predicted query, experimental DB |
| **HA** | holo | AF3 | experimental query, predicted DB (≈ "search an AF2 database with a known target") |
| **AA** | AF3 | AF3 | **headline** — fully predicted, the true deployment case |

**Robustness = degradation from HH.** For each method, `Δ = metric(HH) − metric(AF3-cell)`. The
Phase-5 claim is **learned Δ ≪ frozen Δ** across AH/HA/AA, with learned **≥ frozen absolute** on HH.
Both methods scored on **identical** query/DB pairs and identical patches (`pos_sc` and dense `pos`,
both — Phase 4 §23 showed the sc-gated patch flatters frozen; the dense patch is deployment-real).

## 6. The atom-graph fate — resolved here, on the right test (deferred from Phase 4)

The atom/chem graph was a null across Phases 2–4, but **only ever on the AF3 proxy at small scale or
holo**. The graph's original hypothesis — it encodes connectivity + bond rotatability, so it helps
under **conformation change** — has never been tested where it should matter most. Phase 5 runs the
ablation **surface-only vs surface+atom-graph** on the AF3-involving cells (AH/HA/AA) at full test
scale. **Retire the graph only if it is still null here.** If it measurably reduces AF3 degradation,
that revives the D1-B escalation. Either outcome is a clean, publishable result.

## 7. Milestones (cheapest-first, each gated)

- **M0 — split + harness + de-risk subset (CPU + tiny GPU, ~CHF <1).** (a) Build the
  **sequence-cluster-clean test split** (§4: mmseqs2 ~30% id over all 5,902; report the surviving
  cluster-clean eval count). (b) Rebuild surfaces/descriptors for a ~50-complex subset of it;
  generate AF3 for it; confirm **HH reproduces MaSIF-search** (frozen top-5 in the known ~0.6–0.8
  band) and all four cells run end-to-end. Gate: cluster-clean split built, harness validated, no
  leakage, AF3 pipeline produces sane models. **Kill if HH doesn't reproduce.**
- **M1 — AF3 generation at test scale (GPU, the cost gate — needs human GO).** Generate AF3 apo for
  the **cluster-clean** test sides + decoy pool. Staged in waves with a running cost log;
  checkpointed so it can stop/resume. This is the budget driver (§8).
- **M2 — THE GATE (CPU-feasible eval).** Full 959-test retrieval, all four cells, learned vs frozen,
  robustness Δ + per-complex spread + controls. Decides §3.
- **M3 — graph ablation + write-up.** §6 ablation on AA; consolidate into `10`/a new results doc;
  retire-or-revive the graph; recommend Phase 6 (ligand/neosurface + TED-domainome inference) or a
  redesign if the gate fails.

## 8. Compute & budget (the AF3 driver — gate hard)

- **AF3 generation dominates.** The **cluster-clean** test set (§4) × ~2 sides ≈ up to ~1.9k isolated
  predictions before pruning (fewer after — the cluster filter shrinks the denominator *and* the
  AF3 bill), each = MSA (CPU, can be heavy) + inference (Kuma GPU). Phase 3 ran ~31. **Do not launch
  M1 without an explicit spend estimate + human GO** — first extrapolate cost from the M0 subset and
  the surviving cluster-clean count, then decide full-set vs a curated smaller test panel.
- **Everything else is cheap.** Surface/descriptor precompute is CPU (reference `.sif`). Retrieval
  eval is CPU (encode + score). Encoder training, *if* we conformer-augment (D-P5.1), reuses the
  Phase-4 GPU recipe (~CHF 2–3), gated separately.
- **Budget framing** per CLAUDE.md: the CHF-100 ceiling is a per-session unattended guardrail, not a
  project total. Interactive, human-in-loop work here starts fresh; the real protection is the
  no-GPU-launch-without-GO rule above.

## 9. Risks & traps

1. **AF3 apo isn't apo enough.** AF3 may predict a near-holo conformation for some chains → the
   holo→AF3 gap collapses and the test is toothless. Mitigation: report the **actual** per-complex
   holo↔AF3 backbone/side-chain RMSD distribution; if AF3 sits on top of holo, the robustness claim
   is untestable *and* the deployment concern is moot — say so honestly.
2. **Multi-chain sides.** Isolated AF3 of a multi-chain side ≠ its bound arrangement; interface
   definition needs care. Handle side-as-subcomplex explicitly; drop ill-defined cases with a count.
3. **Homolog leakage (the subtle one).** Complex-level or even PDB-stem disjointness still leaks
   *homologs* — a test partner with a ≥30%-id twin in training isn't truly held out (Stage-C §6).
   Enforce **sequence-cluster** disjointness (§4) and audit the surviving eval clusters; the
   cluster-clean count is the honest denominator, not 959.
4. **Reference-stack fragility at scale.** The `.sif` MSMS/APBS pipeline fails on some PDBs; track a
   success/failure ledger and never silently drop.
5. **Reading single top-k points.** Mandate median-rank + full per-complex spread (Phase 4 §21).
6. **Encoder must be centered.** Any eval off the invariant encoder without DC-offset centering
   collapses to chance (§24) — bake `--center` into the eval harness and assert `z_std` sanity.

## 10. Explicitly NOT in Phase 5

- **Ligands / neosurfaces / molecular-glue** (CRBN+pomalidomide, ligand-restricted query surfaces) →
  **Phase 6**. Needs the ligand-aware surface processing in `masif-neosurf-af2` (**deleted by the
  30-day scratch policy — reclone `git@github.com:meng-yx/masif-neosurf-af2.git -b score_binder`**)
  and likely encoder changes (the Phase-4 encoder ingests a protein atom graph, not ligand atoms).
- **Full TED human-domainome inference** (~25–40k domains) → **Phase 6 deployment**. Inference-only;
  no reliable known-positives; used to *apply* a validated model, not to *validate* it.
- **Experimental apo / Docking Benchmark 5.x** → **never** (wrong domain; deployment queries an
  AI-predicted DB, not experimental unbound crystals — retracted this session).
- **The Stage-C protein-level FP/precision funnel** — OT + spectral-consistency correspondence
  scorer, restraint-guided Boltz-2/Chai co-folding, and the **hard-negative** stress test
  (crystal-contact decoys, swapped-partner, Negatome) — is a **separate later phase** built on
  Phase 5's validated Stage-1 retriever (`docs/11-phase4-stageC-ppi-scoring.md`; renumber into a
  post-Phase-5 slot when it starts). Phase 5 answers *"can we rank the true binder,"* not
  *"do these two proteins bind (with controlled false positives)."*
- **A learned pose scorer / aligner hardening** → later, only if retrieval passes.

## 11. Decisions this phase locks / open D-decisions

- **Locked:** apo = **AI-predicted (AF3)**, not experimental unbound. Validation stays on the
  **PDB list** (holo + self-generated AF3), not the TED domainome. Ligands deferred to P6.
- **D-P5.1 (open) — encoder for M2:** reuse the Phase-4 centered full-set encoder
  (`/work/upthomae/Meng/phase4/ret_full_ctr_best.pt`) **as-is**, or **conformer-augmented retrain**
  (Phase-4 B.1 style: mix AF3 conformers into training). Default: **evaluate the existing encoder
  first** (zero training cost); only retrain if AF3 cells underperform HH by more than frozen's
  robustness margin. Resolve at M2 start.
- **D-P5.2 (open) — test scope if AF3 is too costly:** full 959 vs a curated representative panel
  (~150–200) balanced by interface size/family. Decide from the M0 cost extrapolation.

## 12. Module layout & immediate next actions

Planned (not yet written): `src/masif_graph/p5/{cluster_split.py, af3_apo.py, retrieval_bench.py, cells.py}`,
`scripts/p5_af3_apo_*.sbatch`, `scripts/p5_bench.sbatch`; results → `docs/15-phase5-results.md`,
logs → `logs/phase5/`. **Immediate next actions (all M0, cheap, no GPU launch):**
1. Reclone `masif-neosurf-af2` (needed for surfaces + the training lists).
2. **Build the sequence-cluster-clean split** (mmseqs2 ~30% id over all 5,902 chains; report surviving eval count), then pin the 50-complex M0 subset from the cluster-clean eval set.
3. Stand up the retrieval harness on holo-only and **reproduce MaSIF-search HH** as the sanity gate.
4. Wire AF3 apo generation for the subset; measure per-model cost → the M1 estimate that gates the full run.
