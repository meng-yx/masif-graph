"""Row-preserving graph ablations for Phase-8 A1 (is the Stage-1 encoder sidechain-blind?).

Every transform here keeps the encoder's OUTPUT ROW SET identical: the encoder emits one row per
surface atom (`z = readout(ha[surf_node_idx])`), and the Phase-5 retrieval bench indexes those rows
through the holo/AF3 intersection (`Rec.inter`). Deleting atom nodes would renumber `surf_node_idx`
and silently invalidate that index — so we never delete nodes. We cut *edges* and destroy *features*
instead, which is what "how much does the output depend on sidechains?" actually asks.

Deviation from `docs/24` §3.1, deliberately: the plan proposed rebuilding graphs with backbone-only
atom nodes. That changes the row set (many surface atoms ARE sidechain atoms), so the ablated and
intact runs would retrieve over different patches and the comparison would be confounded by patch
size. The ladder below is the row-matched version of the same question.

Atom feature layout is `p6.atoms` (D=26): col 10 = is_ligand, col 11 = is_backbone, col 22 =
flex_depth/8.

Ablations
  none        control — reproduce the published number (a reproduction check, not a result)
  sc_feat     sidechain atom features permuted among sidechain atoms   (kills sidechain CHEMISTRY)
  sc_edge     aa edges incident to any sidechain atom removed          (kills sidechain CONNECTIVITY)
  sc_all      sc_feat + sc_edge + va edges into sidechain atoms cut    (sidechain atoms isolated)
  bb_feat     backbone atom features permuted                          (comparison channel)
  vert_feat   vertex features permuted                                 (the surface chemistry channel)
  all_feat    atom AND vertex features permuted   <-- POSITIVE CONTROL: this MUST collapse.

`all_feat` is the guardrail. If retrieval survives the destruction of every node feature in the
graph, the harness is not measuring what it claims to and no "sidechain-blind" conclusion may be
drawn from the other rows.
"""
from __future__ import annotations

import numpy as np
import torch

COL_IS_LIGAND = 10
COL_IS_BACKBONE = 11
COL_FLEX_DEPTH = 22

ABLATIONS = ("none", "sc_feat", "sc_edge", "sc_all", "bb_feat", "vert_feat", "all_feat")


def sidechain_mask(g) -> torch.Tensor:
    """Protein sidechain atoms: not backbone and not ligand. (Eval graphs are PPI, so no ligand.)"""
    af = g["atom_feat"]
    return (af[:, COL_IS_BACKBONE] < 0.5) & (af[:, COL_IS_LIGAND] < 0.5)


def backbone_mask(g) -> torch.Tensor:
    af = g["atom_feat"]
    return (af[:, COL_IS_BACKBONE] >= 0.5) & (af[:, COL_IS_LIGAND] < 0.5)


def _permute_rows(feat: torch.Tensor, mask: torch.Tensor, rng) -> torch.Tensor:
    """Permute the masked rows of `feat` among themselves. Returns a new tensor."""
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    if idx.numel() < 2:
        return feat.clone()
    perm = torch.as_tensor(rng.permutation(idx.numel()), dtype=torch.long, device=feat.device)
    out = feat.clone()
    out[idx] = feat[idx[perm]]
    return out


def _cut_aa(g, mask) -> dict:
    """Drop atom-atom edges with either endpoint in `mask`."""
    e = g["aa_edge"]
    if e.shape[1] == 0:
        return {}
    keep = ~(mask[e[0]] | mask[e[1]])
    return {"aa_edge": e[:, keep], "aa_feat": g["aa_feat"][keep]}


def _cut_va(g, mask) -> dict:
    """Drop vertex-atom edges whose atom endpoint is in `mask` (kills both va and av directions:
    the layer builds the atom->vertex message from the same index arrays)."""
    if g["va_a"].shape[0] == 0:
        return {}
    keep = ~mask[g["va_a"]]
    return {"va_v": g["va_v"][keep], "va_a": g["va_a"][keep], "va_feat": g["va_feat"][keep]}


def ablate_graph(g: dict, kind: str, seed: int = 0) -> dict:
    """Return a NEW graph dict with `kind` applied. `g` is never mutated."""
    if kind == "none":
        return g
    if kind not in ABLATIONS:
        raise ValueError(f"unknown ablation {kind!r}; expected one of {ABLATIONS}")
    rng = np.random.default_rng(seed)
    out = dict(g)
    sc = sidechain_mask(g)

    if kind in ("sc_feat", "sc_all"):
        out["atom_feat"] = _permute_rows(g["atom_feat"], sc, rng)
    if kind == "bb_feat":
        out["atom_feat"] = _permute_rows(g["atom_feat"], backbone_mask(g), rng)
    if kind in ("vert_feat", "all_feat"):
        allv = torch.ones(g["vert_feat"].shape[0], dtype=torch.bool, device=g["vert_feat"].device)
        out["vert_feat"] = _permute_rows(g["vert_feat"], allv, rng)
    if kind == "all_feat":
        alla = torch.ones(g["atom_feat"].shape[0], dtype=torch.bool, device=g["atom_feat"].device)
        out["atom_feat"] = _permute_rows(g["atom_feat"], alla, rng)
    if kind in ("sc_edge", "sc_all"):
        out.update(_cut_aa(g, sc))
    if kind == "sc_all":
        out.update(_cut_va(g, sc))
    return out


def ablate_rec(rec, kind: str, seed: int = 0) -> None:
    """Apply `kind` in place to every graph a `p4.eval_af3.Rec` holds (holo + AF3, both chains)."""
    for i, attr in enumerate(("hg1", "hg2", "ag1", "ag2")):
        g = getattr(rec, attr, None)
        if g is not None:
            setattr(rec, attr, ablate_graph(g, kind, seed=seed * 977 + i))


def ablation_stats(g: dict, kind: str, seed: int = 0) -> dict:
    """What the ablation actually removed — reported so a null cannot be blamed on a no-op."""
    ab = ablate_graph(g, kind, seed=seed)
    sc = sidechain_mask(g)
    return {
        "n_atom": int(g["atom_feat"].shape[0]),
        "n_sidechain_atom": int(sc.sum()),
        "n_surf": int(g["surf_node_idx"].shape[0]),
        "n_surf_sidechain": int(sc[g["surf_node_idx"]].sum()),
        "aa_edges_before": int(g["aa_edge"].shape[1]),
        "aa_edges_after": int(ab["aa_edge"].shape[1]),
        "va_edges_before": int(g["va_a"].shape[0]),
        "va_edges_after": int(ab["va_a"].shape[0]),
        "atom_feat_rows_changed": int((ab["atom_feat"] != g["atom_feat"]).any(1).sum()),
        "vert_feat_rows_changed": int((ab["vert_feat"] != g["vert_feat"]).any(1).sum()),
    }
