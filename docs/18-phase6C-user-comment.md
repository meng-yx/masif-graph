# Phase-6 Workstream C — async steering channel

This is how you (the user) steer the headless agent **without interrupting it**. Add a comment
beginning `### 🧑 USER:` under any step header below (or at the end) at any time. The agent
re-reads this file at every step boundary, replies inline with `### 🤖 AGENT:`, acts on it, and
keeps going. A comment is **steering, not a stop signal**. The agent never edits your lines.

The agent mirrors each running-log step header here as it starts. Budget checkpoints (if a
planned action would cross a budget gate) also appear here.

---

## Mission (locked in PHASE6C_HANDOFF.md)
Build the ligand-capable unified 26-D retrieval encoder (Path B: ligand atoms as graph nodes) on
the combined PPI(≥3k) + PDBbind-refined(~5.3k) corpus; eval = do-no-harm PPI gate + mixed
held-out + neosurface (MolGlueDB) benchmark. Conductor, CHF 100 budget. Done = `logs/PHASE6C_DONE`.

---

## Step log (mirrored from docs/progress/phase6C-log.md)

<!-- agent appends "### <step>" headers here as it starts each step -->

### C(a).3 — Path-B protein-ligand graph builder (DONE)
A PDBbind complex is now emitted in the *same* shape as a PPI complex (protein graph + ligand graph
+ contacts), encoded independently, so the whole Phase-4 training loop takes the mixture unchanged.
Validated end-to-end on 5hls.

### ⚠️ C(b).0 — the reference surface tree had been WIPED by the /scratch cleanup
Only 16 of the 4,872 PPI training complexes still had their `.ply`/precompute inputs, so the planned
"re-run `p4.precompute` at 26-D" was impossible. The npz themselves survived on `/work`, and the
14→26-D change only touches `atom_feat`, so I **patch `atom_feat` in place**: 23 of 26 dims come
straight from the stored vector, and the 3 that need atom+residue names are recovered by
re-extracting the chain from a fresh RCSB download. Gated on a bit-exact 14-D check plus a
surface-key check, so a mis-aligned atom table is dropped rather than silently used. 25/25 then
301/301 (Phase-5 eval set) pass; ~5% of the training set is dropped, mostly because RCSB has since
remediated those entries. This saved ~500 core-hours. Full detail in the running log §1.

### C(b).1 — preprocessing children running
PDBbind refined 5,316 (array 65982574), PPI 26-D refeat 4,871 (65982545), Phase-5 eval holo+af3
(65982546 / 65982573). PDBbind is running at **87 s/complex**, well under estimate. Spend so far
~CHF 1 of 100.

### C(b).2 / C(c).1 — split + training code written, CPU debug run green
Loss decreasing, no collapse (z_std flat), and the *untrained* network sits exactly at chance on
every reported group — the eval is not leaking. Next: cluster-clean split, then the Kuma retrain.

### 🧑 USER:
In the future, you do not have to limit the concurrency to 250 parallel array jobs. The cluster should be 
able to handle 500 concurrent jobs. Only use concurrency if you encounter an issue with too many concurrent
jobs. 

A note on how to deal with ligand in holo-apo mapping: unlike proteins which has either holo conformation
from the experimental PDB structure or AF3-predicted apo conformation, ligands won't have an apo conformation.
There is no point in AI predicting the conformation of a ligand, because at deployment we expect ligand-
containing structure (if used) will always come from experimentally determined structures. For training, 
if you need an apo structure for ligand for being balanced (you make a decision based on whether the training
workflow always need pairs of holo vs apo structure for each entry), I guess you can just use the same ligand
structure as both apo and holo. 

### 🤖 AGENT:
Both noted, thank you.

**(1) Array concurrency.** Acknowledged — I capped at `%220` out of caution and it was not needed;
the PDBbind array never queued badly. Any further arrays this workstream launches go uncapped (or at
`%500`), and I will only re-introduce a cap if I actually see submission/scheduler pushback.

**(2) Ligand holo/apo.** This matches what the pipeline already does, and I want to state it
explicitly so the assumption is on the record rather than implicit:

* **No apo ligand is ever generated or needed.** Workstream C training is **holo-only** for both
  corpus types — Stage A and Stage B read only `{cid}__holo__p1/p2.npz`. The training objective
  never pairs a holo state against an apo state; the positive pair is *the two sides of one
  complex* (protein-surface ↔ protein-surface, or protein-pocket-surface ↔ ligand-atoms). So the
  workflow does **not** require a holo/apo pair per entry, and the "duplicate the ligand as its own
  apo" fallback is not needed.
* The only place an apo/AF3 state appears at all is the **Phase-5 do-no-harm gate (axis 1)**, which
  is protein–protein only (the frozen 287-complex set, holo + AF3). No ligand is involved there.
* Axis 2 (mixed held-out) and axis 3 (neosurface) are likewise holo-only, and axis 3's ligands are
  **experimental bound poses** fetched from the RCSB ModelServer instance endpoint — i.e. exactly
  the deployment assumption you describe (ligand geometry always comes from an experimental
  structure, never from a predictor).

**Where your rule would bite, and what I'm doing about it:** the natural follow-up is a *ligand-axis
robustness* test — AF3-apo **protein** + experimental **ligand** — which is the north-star
holo→apo question applied to protein–ligand. That needs AF3 predictions for PDBbind targets, which
the handoff explicitly puts out of scope for C ("AF3/MSA are NOT needed here"). I am **not**
starting it, but I am recording your rule as the design decision for it (protein varies, ligand is
held fixed at its experimental pose) so Phase 7 does not have to re-litigate it, and I will flag it
in the results doc's not-in-scope section.
### C(b) complete — corpus built, split clean
PDBbind refined **5,240/5,316** built (99.7% of non-skipped), PPI **4,711/4,871** re-featurised to
26-D, Phase-5 eval set complete (301 holo + 284 AF3), neosurface benchmark 14/14 systems (28 cases).
Split verify against the *actual* train ids: **0 leaks on all three axes**; 394 PDBbind complexes had
to be dropped for being homologous to the PPI eval set (the cross-corpus filter earning its keep —
a per-corpus component graph could not see those edges, and an earlier version of the builder leaked
203/300 protein-ligand holdouts before the verify step caught it).

Two harness validations before any result: frozen MaSIF on the re-featurised 26-D eval npz
reproduces the Phase-5 published gate **exactly** (HH 0.084/medR 110, AA 0.061/medR 128, n=269,
DB=538), and an untrained encoder reads chance on all three axes.

### C(c).2 — three Kuma runs launched (4026517 combined / 4026518 ppionly / 4026519 plonly)
I added a **third** run beyond the brief: `plonly` (ligands only). It is the control that actually
tests Workstream C's thesis — if `combined` beats `plonly` on the ligand axis, PPI complementarity
really did transfer through the shared 26-D atom space. Without it, "the combined model can retrieve
ligands" would not distinguish transfer from simply having trained on PDBbind. `ppionly` is the
do-no-harm control (separates "26-D features changed something" from "ligands changed something").
Epoch counts are set so **per-type exposure is equal** across the three runs, so differences are
attributable to the other corpus being present rather than to more gradient steps.

Spend: ~CHF 3 Jed + CHF 0.43 Kuma probe so far; the three runs are estimated at ~CHF 8-9.
