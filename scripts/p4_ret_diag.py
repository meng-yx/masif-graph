"""Diagnose the flat retrieval loss: decompose the objective + measure chain-score separation.

Loads the SAME init checkpoint + a batch of the SAME train complexes the proof used, then reports:
  1. per-component loss (chain-CE vs log(N) floor, atom-InfoNCE, vicreg var/cov) at init
  2. normalized interface-patch collinearity: per-dim std vs the uniform-sphere ideal 1/sqrt(d),
     mean off-diagonal cosine of patch centroids  -> is the encoder's *direction* space collapsed?
  3. chain score matrix M: does the true partner outscore decoys? (argmax-partner accuracy, and the
     partner-vs-mean-decoy margin in units of tau_c) -> is there ANY signal for CE to amplify?
  4. gradient norm into the encoder from the chain term alone -> is the objective even pushing?
"""
import math, os, sys
import numpy as np
import torch
torch.set_num_threads(8)

from masif_graph.p4.encoder import HeteroEncoder
from masif_graph.p4.objective import (Complementarity, normalize, vicreg_terms,
                                       info_nce_complex, chain_retrieval_loss, chain_patch_score_matrix)
from masif_graph.p4.dataset import ComplexP4, usable_complexes, D_AA, D_VV, D_VA

DATA = "logs/phase4/m2_npz"
IDS = "logs/phase4/m2_train_ids.txt"
CKPT = "/work/upthomae/Meng/phase4/vicreg_sc_best_seed0.pt"
POS = "pos"   # dense patch, as in the proof
NB = 24
TAU_C, TAU_ATOM = 0.07, 0.1

dev = "cpu"
ids = usable_complexes(DATA, [l.strip() for l in open(IDS) if l.strip()])[:NB]
cs = [ComplexP4(DATA, c, dev) for c in ids]
def iface(pos, col): return torch.unique(pos[:, col]) if pos.shape[0] else torch.zeros(0, dtype=torch.long)
cs = [c for c in cs if iface(getattr(c, POS), 0).numel() and iface(getattr(c, POS), 1).numel()]
print(f"loaded {len(cs)} complexes -> {2*len(cs)} chains")

fa, fv = cs[0].p1["atom_feat"].shape[1], cs[0].p1["vert_feat"].shape[1]
enc = HeteroEncoder(fa, fv, D_AA, D_VV, D_VA, d=64, d_out=32, n_layers=4).to(dev)
comp = Complementarity(32, tau_init=0.1).to(dev)
ck = torch.load(CKPT, map_location=dev)
enc.load_state_dict(ck["enc"]); comp.load_state_dict(ck["comp"])
with torch.no_grad(): comp.log_tau.fill_(math.log(0.1))
enc.train()

patches, raws, partner = [], [], []
zfull = []
for c in cs:
    z1r, z2r = enc(c.p1), enc(c.p2)
    z1n, z2n = normalize(z1r), normalize(z2r)
    i1, i2 = iface(getattr(c, POS), 0), iface(getattr(c, POS), 1)
    raws += [z1r[i1], z2r[i2]]
    patches += [z1n[i1], z2n[i2]]
    b = len(patches); partner += [b-1, b-2]
    zfull.append((z1n, z2n, getattr(c, POS)))
partner = torch.tensor(partner)
N = len(patches)
print(f"N={N} chain patches; sizes min/med/max = "
      f"{min(p.shape[0] for p in patches)}/{int(np.median([p.shape[0] for p in patches]))}/{max(p.shape[0] for p in patches)}")

# --- (2) collapse of the normalized direction space ---
allz = torch.cat([p.detach() for p in patches], 0)
perdim = allz.std(0).mean().item()
ideal = 1/math.sqrt(32)
cents = torch.stack([normalize(p.detach().mean(0, keepdim=True))[0] for p in patches])  # (N,32)
cos = cents @ cents.t()
offcos = cos[~torch.eye(N, dtype=bool)].mean().item()
print(f"\n[direction collapse] normalized per-dim std = {perdim:.4f}  (uniform-sphere ideal {ideal:.4f}; "
      f"ratio {perdim/ideal:.2f})")
print(f"[direction collapse] mean off-diag cosine of chain centroids = {offcos:.3f}  (0=orthogonal, 1=identical)")

# --- (3) chain score matrix separation ---
with torch.no_grad():
    M = chain_patch_score_matrix([p.detach() for p in patches], comp, TAU_ATOM)  # (N,N) raw scores
Mm = M.clone(); Mm.fill_diagonal_(-1e9)
pred = Mm.argmax(1)
acc = (pred == partner).float().mean().item()
# partner score vs mean decoy score, per chain, in units of tau_c
psc = M[torch.arange(N), partner]
mask = torch.ones(N, N, dtype=bool); mask[torch.arange(N), torch.arange(N)] = False
mask[torch.arange(N), partner] = False
dec_mean = (M.masked_fill(~mask, 0).sum(1) / mask.sum(1))
margin = ((psc - dec_mean) / TAU_C)
print(f"\n[chain separation] argmax-partner accuracy = {acc:.3f}  (chance {1/(N-1):.3f})")
print(f"[chain separation] (partner - mean_decoy)/tau_c  mean={margin.mean():+.3f}  "
      f"median={margin.median():+.3f}  (want >>0; this is the CE logit gap)")
print(f"[chain separation] raw M off-diag range: min={M[mask].min():.3f} max={M[mask].max():.3f} "
      f"std={M[mask].std():.4f}")

# --- (1) loss decomposition ---
chain = chain_retrieval_loss([p for p in patches], partner, comp, tau_c=TAU_C, tau_atom=TAU_ATOM)
al, na = 0.0, 0
for (z1n, z2n, tp) in zfull:
    if tp.shape[0] > 0: al = al + info_nce_complex(z1n, z2n, tp, comp); na += 1
atom = al/na
zc = torch.cat(raws, 0)
v, cc = vicreg_terms(zc)
print(f"\n[loss decomp] chain-CE      = {float(chain):.3f}   (uniform-softmax floor log(N-1)={math.log(N-1):.3f})")
print(f"[loss decomp] atom-InfoNCE  = {float(atom):.3f}   x w_atom 0.5 = {0.5*float(atom):.3f}")
print(f"[loss decomp] vicreg var    = {float(v):.3f}   x 2.0 = {2.0*float(v):.3f}")
print(f"[loss decomp] vicreg cov    = {float(cc):.3f}   x 0.04 = {0.04*float(cc):.3f}")
tot = chain + 0.5*atom + 2.0*v + 0.04*cc
print(f"[loss decomp] TOTAL         = {float(tot):.3f}   (proof reported ~6.8)")

# --- (4) gradient magnitude from the chain term alone ---
enc.zero_grad(); comp.zero_grad()
chain.backward()
gnorm_enc = math.sqrt(sum((p.grad**2).sum().item() for p in enc.parameters() if p.grad is not None))
gnorm_T = math.sqrt(sum((p.grad**2).sum().item() for p in comp.parameters() if p.grad is not None))
print(f"\n[grad] ||d chain-CE / d enc|| = {gnorm_enc:.4e}   ||d/dT|| = {gnorm_T:.4e}")
print(f"[grad] ||T||_fro = {comp.T.norm().item():.3f}   tau(frozen) = {comp.log_tau.exp().item():.3f}")

# --- (5) DECISIVE: is the direction-collapse a removable DC offset, or intrinsic? ---
print("\n=== centering test: subtract global mean of raw patch atoms, then re-normalize ===")
raw_cat = torch.cat([r.detach() for r in raws], 0)          # all interface atoms, raw
gmean = raw_cat.mean(0, keepdim=True)                        # (1,32) common offset
cents_raw = torch.stack([r.detach().mean(0) for r in raws])  # (N,32) per-chain raw centroid
spread = (cents_raw - gmean).norm(dim=1).mean().item()
print(f"||global mean|| = {gmean.norm().item():.3f}   mean ||centroid - gmean|| = {spread:.3f}   "
      f"ratio {gmean.norm().item()/spread:.1f}x  (>>1 => DC offset dominates)")
# recompute centroid cosine AFTER centering
cc2 = torch.stack([torch.nn.functional.normalize((r.detach()-gmean).mean(0, keepdim=True))[0] for r in raws])
cos2 = cc2 @ cc2.t()
off2 = cos2[~torch.eye(N, dtype=bool)].mean().item()
print(f"centroid cosine  before centering = {offcos:.3f}   AFTER centering = {off2:.3f}")
# rebuild chain score matrix on CENTERED+renormalized patches, check partner separation
cpatch = [torch.nn.functional.normalize(r.detach()-gmean) for r in raws]
with torch.no_grad():
    M2 = chain_patch_score_matrix(cpatch, comp, TAU_ATOM)
M2m = M2.clone(); M2m.fill_diagonal_(-1e9)
acc2 = (M2m.argmax(1) == partner).float().mean().item()
psc2 = M2[torch.arange(N), partner]
dec2 = (M2.masked_fill(~mask, 0).sum(1)/mask.sum(1))
mar2 = ((psc2-dec2)/TAU_C)
print(f"argmax-partner acc  before = {acc:.3f}   AFTER centering = {acc2:.3f}   (chance {1/(N-1):.3f})")
print(f"(partner-mean_decoy)/tau_c  before mean={margin.mean():+.3f}   AFTER mean={mar2.mean():+.3f} median={mar2.median():+.3f}")
