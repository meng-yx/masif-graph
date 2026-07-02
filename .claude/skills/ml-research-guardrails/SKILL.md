---
name: ml-research-guardrails
description: >-
  Self-audit protocol for an agent doing autonomous machine-learning / neural-network
  research on an HPC cluster (SLURM). Use continuously during long unattended runs to
  guard against the high-risk SILENT failure modes: data leakage, inflated/too-good
  metrics, missing baselines, context decay over long horizons, and unfounded
  conclusions. Trigger whenever the agent is training models, building datasets,
  splitting data, evaluating results, or is about to report a result or move to the
  next experiment. The guiding rule: a crash is cheap, a confident wrong result is
  expensive — so actively try to disprove your own good news before believing it.
---

# ML Research Guardrails

You are running with limited human supervision. Your most dangerous failure is not a
crash — it is producing a result that looks like success but is scientifically wrong
(a leaky split, a memorizable negative set, a metric bug), and then reporting it as a
win. This skill forces you to interrogate your own work before trusting it.

Adopt an adversarial stance toward your own good results. If a number looks great, your
first assumption is that something is broken, not that you succeeded. Only after you
have actively tried and failed to break it may you believe it.

## When to run these checks

Do not run these only at the end. Run the relevant section at each of these moments,
and write the result to your progress log (see "Checkpointing"):

- After building or splitting a dataset — before any training. (Leakage + negatives)
- After the first training run finishes — before launching a sweep. (Plausibility + baselines)
- Before reporting ANY metric to the human or recording it as a result. (Plausibility + honesty)
- Every ~1–2 hours of wall-clock, or every ~30 tool calls. (Coherence)
- Before concluding an experiment or picking a "best" model. (All sections)

## Section 1 — Data leakage (the #1 killer)

Leakage is when information from the test set reaches the model during training. It is
the single most common cause of impressive-but-fake results. Answer each in writing;
do not answer from memory, verify against the actual code/data:

1. How exactly are train/val/test split? Print the split code. Is it a naive random
   split? If so, STOP — random splits leak in most biological and relational data.
2. Are near-duplicate or highly similar examples able to land on opposite sides of the
   split? For sequences/molecules/graphs, splits must be by *identity/similarity
   cluster* (e.g. sequence-identity clustering, scaffold split, time split), not by row.
   Confirm the clustering actually ran and reduced cross-split similarity.
3. Was any preprocessing (normalization stats, feature selection, vocabulary,
   embeddings, PCA, imputation, target encoding) fit on the FULL dataset before
   splitting? It must be fit on train only, then applied to val/test.
4. Do any features encode the label directly or by proxy (IDs, timestamps, source
   database, row order, class-correlated metadata)? Check feature importances for a
   single feature dominating — that is a red flag.
5. For paired/relational data (e.g. interaction prediction): can the model win by
   memorizing node frequency or degree instead of learning the relation? Are negatives
   constructed so they are not trivially separable from positives (matched degree,
   shared distribution, hard negatives)?
6. Is the test set touched during model selection? Model/hyperparameter choice must use
   val only; test is looked at once, at the end.

If you cannot affirmatively clear all six, treat the current results as invalid.

## Section 2 — Result plausibility ("too good to be true")

Before believing any metric:

1. Compare against a naive baseline you actually ran: random guessing, majority class,
   single-strongest-feature, and a shuffled-label control. A shuffled-label run MUST
   collapse to chance — if it doesn't, you have leakage or a metric bug. Run it.
2. Is the score suspiciously high for this problem? Check literature/SOTA for the same
   benchmark. Beating SOTA by a large margin on the first serious attempt is almost
   always a bug, not a breakthrough.
3. Re-derive the metric on a tiny hand-checkable subset. Confirm the metric code
   matches its definition (e.g. AUC vs accuracy vs AUPRC under class imbalance — report
   AUPRC when positives are rare).
4. Is val/test performance wildly better than train? That is usually a data or metric
   error, not generalization.
5. Does performance hold on a second, independently constructed test set? If you only
   have one, say so and lower your confidence accordingly.

## Section 3 — Baseline and experimental discipline

1. Every reported improvement must be relative to a baseline you ran under identical
   conditions, not a remembered number from a paper.
2. Change one thing at a time. If you changed the data and the model and the metric
   between two runs, you cannot attribute the difference.
3. Report variance: run key comparisons with ≥3 seeds and give mean ± std. A single-seed
   win inside the noise band is not a win.
4. Keep the exact command, config, git commit / code hash, and dataset version for every
   result. If you can't reproduce it, it isn't a result.

## Section 4 — Context and coherence decay (long-horizon risk)

Over many hours you will lose the thread. Counteract it:

1. Re-read the original design goal and your progress log at the start of each work
   cycle. Confirm your current action still serves the goal; if it has drifted, stop
   and correct.
2. Check for repeated or contradictory work: are you re-running something you already
   did, or reversing an earlier decision without noting why?
3. If SLURM jobs are queued/idle, do not busy-poll and burn context. Record the job IDs,
   the expected completion signal, and what you will do with the result, then stop or
   back off with longer sleeps.
4. If you notice your own confusion, summarize state to the log and reset to a known-good
   checkpoint rather than pushing forward.

## Section 5 — Honest reporting and stop conditions

1. Report uncertainty explicitly. State what was NOT tested and what could still be
   wrong. Never upgrade "looks promising" to "works."
2. Distinguish "the pipeline ran without errors" from "the result is scientifically
   valid." They are different claims; make only the one you've earned.
3. HALT and ask the human when:
   - A result beats the baseline or SOTA by a margin large enough to be implausible.
   - You cannot rule out leakage after Section 1.
   - The shuffled-label control does not collapse to chance.
   - You are about to spend significant compute on a direction justified by an unverified
     good result.
   - You have drifted from the design goal and are unsure how to reconcile.

A halt with a clear question is a success, not a failure. Burning six hours of GPU on a
leaky model is the actual failure.

## Checkpointing (do this the whole time)

Maintain a single append-only progress log file (e.g. `PROGRESS.md` in the run
directory). After every meaningful step write: timestamp, what you did, the command /
job ID, the result, which guardrail checks you ran and their outcome, and the next
planned step. This log is what lets you survive a context reset, lets the human audit
you, and lets you notice your own drift. Treat "did I update the log?" as part of
finishing any step.

## One-line creed

Try to break your own good news before you believe it; a crash is cheap, a confident
wrong answer is expensive.