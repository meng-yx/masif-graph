"""Overfit diagnostic: can the chain-retrieval objective fit a tiny train set at all?

The proof logged only the TOTAL loss (chain-CE + 0.5*atom-InfoNCE + VICReg), which is dominated
by the atom-InfoNCE (~ln(900)=6.8) so the chain-CE (max ln(N)) was invisible. This isolates the
chain-CE on ~8 complexes (16 chains = one batch), applies centering (docs/10 §21: without it the
embeddings collapse to cosine 0.999), and logs per-step: chain-CE, TRAIN top-1 retrieval accuracy
(argmax over decoys == true partner), and grad norms. Decisive fork:
  CE -> ~0 & acc -> 100%  => objective CAN fit; proof failure was config/generalization -> GPU worth it
  CE stuck ~ln(N) & acc ~random => objective can't separate partners -> learned-retrieval NO-GO
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "16")
import numpy as np
import torch

from masif_graph.p4.encoder import HeteroEncoder
from masif_graph.p4.objective import Complementarity, normalize, chain_retrieval_loss, chain_patch_score_matrix
from masif_graph.p4.dataset import ComplexP4, usable_complexes, D_AA, D_VV, D_VA

DATA = "logs/phase4/m2_npz"
IDS = "logs/phase4/m2_train_ids.txt"
CKPT = "/work/upthomae/Meng/phase4/vicreg_sc_best_seed0.pt"
N_COMPLEX = 8
EPOCHS = 200
LR = 1e-3
TAU_C = 0.07
TAU_ATOM = 0.1
CENTER = True
dev = "cpu"


def iface_idx(pos, col):
    return torch.unique(pos[:, col]) if pos.shape[0] else torch.zeros(0, dtype=torch.long)


def main():
    ids = usable_complexes(DATA, [l.strip() for l in open(IDS) if l.strip()][:20])
    cs = [ComplexP4(DATA, c, "cpu") for c in ids]
    cs = [c for c in cs if iface_idx(c.pos, 0).numel() > 0 and iface_idx(c.pos, 1).numel() > 0]
    cs = cs[:N_COMPLEX]
    print(f"overfit set: {len(cs)} complexes / {2*len(cs)} chains", flush=True)

    f_atom = cs[0].p1["atom_feat"].shape[1]; f_vert = cs[0].p1["vert_feat"].shape[1]
    enc = HeteroEncoder(f_atom, f_vert, D_AA, D_VV, D_VA, d=64, d_out=32, n_layers=4).to(dev)
    comp = Complementarity(32, tau_init=0.1).to(dev)
    ck = torch.load(CKPT, map_location=dev)
    enc.load_state_dict(ck["enc"]); comp.load_state_dict(ck["comp"])
    with torch.no_grad():
        comp.log_tau.fill_(float(np.log(0.1)))
    comp.log_tau.requires_grad_(False)
    print(f"init from {CKPT}", flush=True)

    opt = torch.optim.Adam([{"params": enc.parameters()}, {"params": comp.parameters()}], lr=LR)

    torch.manual_seed(0)
    for ep in range(EPOCHS):
        enc.train()
        raws, partner = [], []
        for c in cs:
            c = c.to(dev)
            z1r, z2r = enc(c.p1), enc(c.p2)
            i1 = iface_idx(c.pos, 0); i2 = iface_idx(c.pos, 1)
            raws.append(z1r[i1]); raws.append(z2r[i2])
            b = len(raws); partner += [b - 1, b - 2]
        if CENTER:
            mu = torch.cat(raws, 0).mean(0, keepdim=True)
            patches = [normalize(r - mu) for r in raws]
        else:
            patches = [normalize(r) for r in raws]
        partner_t = torch.tensor(partner, device=dev)
        loss = chain_retrieval_loss(patches, partner_t, comp, tau_c=TAU_C, tau_atom=TAU_ATOM)
        opt.zero_grad(); loss.backward()
        gnorm_enc = torch.sqrt(sum((p.grad**2).sum() for p in enc.parameters() if p.grad is not None))
        gnorm_T = torch.sqrt(sum((p.grad**2).sum() for p in comp.parameters() if p.grad is not None))
        opt.step()

        if ep % 10 == 0 or ep == EPOCHS - 1:
            with torch.no_grad():
                M = chain_patch_score_matrix(patches, comp, TAU_ATOM)
                Mm = M - torch.diag(torch.full((M.shape[0],), float("inf")))
                acc = float((Mm.argmax(1) == partner_t).float().mean())
                zc = torch.cat(patches, 0)
                zstd = float(zc.std(0).mean())
            print(f"[ep {ep:3d}] CE={float(loss):.4f}  train_top1={acc:.3f}  "
                  f"g_enc={float(gnorm_enc):.3e} g_T={float(gnorm_T):.3e} zstd={zstd:.4f}", flush=True)
    print(f"random-chance top1 (1/{2*len(cs)-1}) = {1/(2*len(cs)-1):.3f}", flush=True)


if __name__ == "__main__":
    main()
