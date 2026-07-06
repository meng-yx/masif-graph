"""Contrastive trainer + rotamer augmentation + ablation harness + differential evaluation.

The scientific object (design docs/03 §5): for each ablation cell, descriptor-separation AUC on
HOLO (do-no-harm floor) and on the apo-like REPACK, and the *differential degradation*
(holo AUC - repack AUC) relative to surface-only. Positives are the sc-filtered holo contacts,
evaluated on the SAME intersection pairs in both states (identity-mapped). Negatives reuse the
Phase-1 neg_mix and random-neg schemes.

Fused embeddings (design D3-A, D-P2.3):
    fused1 = normalize(Head(surf_straight ⊕ g))   # p1 target role
    fused2 = normalize(Head(surf_flipped  ⊕ g))   # p2 binder role
scored by 1/L2 like the reference. `g` is the role-independent graph readout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from masif_graph.graph.model import MaSIFGraphModel, AblationConfig
from masif_graph.graph.dataset import graph_to_tensors
from masif_graph.metrics.separation import separation_auc, shuffled_label_auc
from masif_graph.experiments.probe_core import ComplexView, score_complex, random_negative_dists

CELLS = [
    AblationConfig(False, False, False, "surface_only"),
    AblationConfig(True,  False, False, "covalent"),
    AblationConfig(True,  True,  False, "rotatability"),
    AblationConfig(False, False, True,  "spatial"),
    AblationConfig(True,  True,  True,  "full"),
]


# ---------------------------------------------------------------- tensor caching
def _tensor_cache(crec_list, use_rotatable, device):
    """Cache GNN tensors per (complex,state,pid). Keyed by object id of the AtomGraph."""
    cache = {}
    for crec in crec_list:
        for state in ("holo", "repack"):
            recs = crec.holo if state == "holo" else crec.repack
            if recs is None:
                continue
            for pid in ("p1", "p2"):
                g = recs[pid].graph
                cache[id(g)] = graph_to_tensors(g, use_rotatable, device=device)
    return cache


def _chain(crec, pid, state):
    return (crec.holo if state == "holo" else crec.repack)[pid]


def _map_pos(crec, s1_state, s2_state):
    """Holo positives mapped to (row_in_s1, row_in_s2) by atom identity; drop unmapped."""
    h1, h2 = crec.holo["p1"].keys, crec.holo["p2"].keys
    r1 = _chain(crec, "p1", s1_state)
    r2 = _chain(crec, "p2", s2_state)
    out = []
    for i, j in crec.holo_pos:
        ki, kj = h1[i], h2[j]
        a = r1.key2row.get(ki)
        b = r2.key2row.get(kj)
        if a is not None and b is not None:
            out.append((a, b))
    return np.array(out, dtype=np.int64) if out else np.zeros((0, 2), np.int64)


# ---------------------------------------------------------------- embedding
def _embed_chain(model, tcache, rec, role):
    """Fused, L2-normalized embedding for one chain. role='straight'(p1) or 'flipped'(p2)."""
    desc = rec.desc_straight if role == "straight" else rec.desc_flipped
    desc_t = torch.tensor(desc, dtype=torch.float32)
    g = model.graph_readout(tcache[id(rec.graph)])
    fused = model.fuse(desc_t, g)
    return F.normalize(fused, dim=1)


# ---------------------------------------------------------------- training
@dataclass
class HParams:
    steps: int = 300
    lr: float = 1e-3
    margin: float = 1.0
    neg_per_pos: int = 8
    p_aug: float = 0.5          # prob a chain uses its repacked state during training
    weight_decay: float = 1e-5
    seed: int = 0


def train_cell(cfg: AblationConfig, train_recs, hp: HParams, device="cpu", log=lambda s: None):
    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)
    rng = np.random.default_rng(hp.seed)
    model = MaSIFGraphModel(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    tcache = _tensor_cache(train_recs, cfg.use_rotatable, device)
    have_repack = {c.holo_id: (c.repack is not None) for c in train_recs}

    for step in range(hp.steps):
        model.train()
        opt.zero_grad()
        # per-chain augmentation state
        comp = []
        for crec in train_recs:
            s1 = "repack" if (have_repack[crec.holo_id] and rng.random() < hp.p_aug) else "holo"
            s2 = "repack" if (have_repack[crec.holo_id] and rng.random() < hp.p_aug) else "holo"
            pos = _map_pos(crec, s1, s2)
            if len(pos) == 0:
                continue
            r1 = _chain(crec, "p1", s1); r2 = _chain(crec, "p2", s2)
            e1 = _embed_chain(model, tcache, r1, "straight")
            e2 = _embed_chain(model, tcache, r2, "flipped")
            comp.append((crec.holo_id, e1, e2, pos))
        if not comp:
            continue
        # global pool of p2 embeddings for cross-complex negatives
        pool = [(cid, e2) for (cid, _e1, e2, _p) in comp]
        loss = 0.0
        npos_total = 0
        for cid, e1, e2, pos in comp:
            ai = torch.tensor(pos[:, 0]); bj = torch.tensor(pos[:, 1])
            d_pos = (e1[ai] - e2[bj]).norm(dim=1)
            P = len(pos)
            # within-complex negatives: random p2 rows (!= true partner)
            wj = torch.tensor(rng.integers(e2.shape[0], size=(P, hp.neg_per_pos // 2)))
            d_win = (e1[ai].unsqueeze(1) - e2[wj]).norm(dim=2)
            # cross-complex negatives: random other-complex p2 rows
            others = [p for p in pool if p[0] != cid]
            d_cross = None
            if others:
                nk = hp.neg_per_pos - hp.neg_per_pos // 2
                cross_cols = []
                for _ in range(nk):
                    ocid, oe2 = others[int(rng.integers(len(others)))]
                    oj = torch.tensor(rng.integers(oe2.shape[0], size=P))
                    cross_cols.append((e1[ai] - oe2[oj]).norm(dim=1, keepdim=True))
                d_cross = torch.cat(cross_cols, dim=1)
            d_neg = d_win if d_cross is None else torch.cat([d_win, d_cross], dim=1)
            loss = loss + (d_pos ** 2).mean() + F.relu(hp.margin - d_neg).pow(2).mean()
            npos_total += P
        loss = loss / max(len(comp), 1)
        loss.backward()
        opt.step()
        if step % max(hp.steps // 6, 1) == 0 or step == hp.steps - 1:
            log(f"    [{cfg.name}] step {step:4d} loss {float(loss):.4f} (npos {npos_total})")
    return model


# ---------------------------------------------------------------- evaluation
def _views_for_state(model, recs, state, use_rotatable, device="cpu"):
    """Build ComplexView per complex in `state`, using fused embeddings (or raw if model is None).
    Uses the intersection positive set so holo/repack are compared on identical contacts."""
    tcache = None
    if model is not None:
        tcache = _tensor_cache(recs, use_rotatable, device)
    views = []
    cross_pool = []
    for crec in recs:
        if state == "repack" and crec.repack is None:
            continue
        pos = crec.inter_pos_holo if state == "holo" else crec.inter_pos_repack
        if pos is None or len(pos) == 0:
            continue
        r1 = _chain(crec, "p1", state); r2 = _chain(crec, "p2", state)
        if model is None:
            s1 = r1.desc_straight; f2 = r2.desc_flipped
        else:
            model.eval()
            with torch.no_grad():
                s1 = _embed_chain(model, tcache, r1, "straight").cpu().numpy()
                f2 = _embed_chain(model, tcache, r2, "flipped").cpu().numpy()
        v = ComplexView(crec.holo_id, s1, f2, r1.coord, r2.coord, pos,
                        hard_radius=6.0, within_min=5.0)
        views.append(v)
        cross_pool.append((crec.holo_id, f2))
    return views, cross_pool


def eval_state(model, recs, state, use_rotatable, seed=0, device="cpu"):
    """Return pooled + per-complex descriptor-separation AUC (random-neg + neg_mix) for `state`."""
    views, cross_pool = _views_for_state(model, recs, state, use_rotatable, device)
    all_pos_nm, all_neg_nm, all_pos_rn, all_neg_rn = [], [], [], []
    per_complex = []
    for k, v in enumerate(views):
        sc = score_complex(v, cross_pool, seed + k)
        rn = random_negative_dists(v, seed + k)
        auc_nm = separation_auc(sc.pos_dists, sc.neg_dists)
        auc_rn = separation_auc(sc.pos_dists, rn)
        per_complex.append({"cid": v.complex_id, "n_pos": int(sc.n_pos),
                            "auc_negmix": auc_nm, "auc_randneg": auc_rn})
        all_pos_nm.append(sc.pos_dists); all_neg_nm.append(sc.neg_dists)
        all_pos_rn.append(sc.pos_dists); all_neg_rn.append(rn)
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
    pooled_nm = separation_auc(cat(all_pos_nm), cat(all_neg_nm))
    pooled_rn = separation_auc(cat(all_pos_rn), cat(all_neg_rn))
    shuf = shuffled_label_auc(cat(all_pos_rn), cat(all_neg_rn), np.random.default_rng(seed))
    return {
        "state": state, "pooled_negmix": pooled_nm, "pooled_randneg": pooled_rn,
        "shuffled_randneg": shuf, "per_complex": per_complex,
        "n_complexes": len(views),
        "median_randneg": float(np.nanmedian([p["auc_randneg"] for p in per_complex]) if per_complex else np.nan),
        "median_negmix": float(np.nanmedian([p["auc_negmix"] for p in per_complex]) if per_complex else np.nan),
    }
