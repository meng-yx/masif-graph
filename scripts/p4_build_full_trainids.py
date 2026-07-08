#!/usr/bin/env python
"""Build the FULL clean Phase-4 Stage-A training id-list (scale-up).

Guard (mandatory — leakage is the #1 silent failure mode):
  candidates = training.txt complexes that
    (1) are complex_is_available() (desc + precomp + pdb + ply on disk), AND
    (2) do NOT share a 4-char PDB stem with any of the 60 held-out eval complexes
        (stageA_heldout_ids.txt) or the 31 Phase-3 m1 AF3-benchmark complexes (m1_ids.txt).
  Variant filter (AF/RP/AS) is a no-op for training.txt (verified all stems are 4-char PDB
  codes) but is applied defensively anyway.

Held-out (60) and m1 (31) are entirely in testing.txt, so they never appear in training.txt;
the stem guard removes any *same-protein different-chain* training complex that would leak.

Outputs (logs/phase4/):
  stageA_full_candidates.txt  — clean available training ids (train pool for Kuma)
  stageA_full_precompute.txt  — candidates + the 60 held-out ids (everything to build npz for)
Prints a full audit so the guard is inspectable.
"""
import os
import sys

REPO = "/scratch/ymeng/masif-graph"
sys.path.insert(0, os.path.join(REPO, "src"))
from masif_graph.io.reference import complex_is_available  # noqa: E402

LISTS = os.path.join(REPO, "data/lists")
P4 = os.path.join(REPO, "logs/phase4")


def read_ids(path):
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def pdb4(cid: str) -> str:
    """4-char PDB stem: text before first '_', truncated to 4 (catches augmented variants too)."""
    return cid.split("_")[0][:4].upper()


def is_variant(cid: str) -> bool:
    stem = cid.split("_")[0]
    return len(stem) != 4 or stem[4:5].isalpha()  # PDB codes are exactly 4 chars


def main():
    training = read_ids(os.path.join(LISTS, "training.txt"))
    heldout = read_ids(os.path.join(P4, "stageA_heldout_ids.txt"))
    m1 = read_ids(os.path.join(REPO, "logs/phase3/m1_ids.txt"))

    excl_stems = {pdb4(c) for c in heldout} | {pdb4(c) for c in m1}
    print(f"training.txt: {len(training)}  held-out: {len(heldout)}  m1: {len(m1)}")
    print(f"excluded PDB stems (held-out ∪ m1): {len(excl_stems)}")

    n_variant = n_stem = n_unavail = 0
    candidates = []
    guard_passed = []
    for cid in training:
        if is_variant(cid):
            n_variant += 1
            continue
        if pdb4(cid) in excl_stems:
            n_stem += 1
            continue
        guard_passed.append(cid)            # passes leakage guard, regardless of availability
        if complex_is_available(cid):
            candidates.append(cid)          # additionally has descriptors on disk NOW
        else:
            n_unavail += 1

    # Hard leakage assertions (never fabricate a clean set) — on the FULL guard-passed set.
    gp_stems = {pdb4(c) for c in guard_passed}
    leak = gp_stems & excl_stems
    assert not leak, f"LEAK: guard-passed stems overlap held-out/m1: {sorted(leak)[:10]}"
    assert not (set(guard_passed) & set(heldout)), "LEAK: guard-passed ∩ held-out non-empty"
    assert not (set(guard_passed) & set(m1)), "LEAK: guard-passed ∩ m1 non-empty"

    print(f"excluded — variant: {n_variant}  stem-overlap: {n_stem}")
    print(f"GUARD-PASSED training ids (leakage-clean, any availability): {len(guard_passed)}")
    print(f"  of which AVAILABLE now (descriptors on disk): {len(candidates)}  "
          f"(not-yet/never available: {n_unavail})")

    # Reference list: currently-available clean candidates (for the audit / eyeballing).
    cand_path = os.path.join(P4, "stageA_full_candidates.txt")
    with open(cand_path, "w") as fh:
        fh.write("\n".join(candidates) + "\n")
    print(f"wrote {cand_path} ({len(candidates)})")

    # Precompute set = ALL guard-passed training + the 60 held-out. p4.precompute SKIPs any id still
    # unavailable, so feeding the full guard-passed set (built once, run under a SLURM dependency after
    # preprocessing fully drains) captures every complex that ends up available — no straggler top-up.
    # train.py's usable_complexes() then self-filters train/val ids to those with npz actually present.
    precompute = guard_passed + heldout
    pre_path = os.path.join(P4, "stageA_full_precompute.txt")
    with open(pre_path, "w") as fh:
        fh.write("\n".join(precompute) + "\n")
    print(f"wrote {pre_path}  ({len(precompute)} ids = {len(guard_passed)} guard-passed train "
          f"+ {len(heldout)} held-out)")


if __name__ == "__main__":
    main()
