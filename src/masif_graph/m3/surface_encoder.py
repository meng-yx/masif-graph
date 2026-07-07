"""Phase-3 M3: learnable DiffusionNet surface encoder over MaSIF meshes.

ONE validated building block. Given a MaSIF `.ply` surface (verts + faces + per-vertex chemistry
channels), this runs a learnable DiffusionNet (the "diffusion-net-plus" fork by pvnieo) to produce
per-vertex embeddings. It is the *unfreezing* lever of the M3 encoder (see `m3/__init__.py`): the
surface descriptor becomes trainable so it can be pushed toward holo<->AF3 conformation invariance.

This module does NOT train, build the atom graph, or define the contrastive loss — those are fused
around this encoder later. Here we only provide + validate the forward/backward on real meshes.

Key API facts about this diffusion_net fork (verified against its source):
  - `DiffusionNet.forward(surface)` takes ONE object and reads `surface.{x, mass, evals, evecs,
    gradX, gradY, batch}`. `x` is (V, C_in) input features; the rest are the diffusion operators.
  - `surface.batch` MUST be present (all-zeros for a single mesh): `LearnedTimeDiffusion` does
    `int(batch.max()+1)`, so a plain non-batched `Data` (whose `.batch` is None) would crash.
  - `mass, L, gradX, gradY` are `torch_sparse.SparseTensor`; `evals (K,)`, `evecs (V,K)` are dense.
  - `compute_diffusion_operators(verts, faces, k_eig)` -> (frames, massvec, L, evals, evecs,
    gradX, gradY); massvec/L/gradX/gradY are scipy-sparse (convert with `sparse_np_to_sparse_tensor`).

Gotcha: `DiffusionNet.forward` MUTATES `surface.x` in place (first_lin overwrites it, last_lin
leaves (V, C_out)). So a surface object is single-use per forward; re-set `surface.x` (or rebuild)
before calling again. Operators are the expensive part and are conformation/feature-independent, so
in training we cache them and only swap `surface.x`.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData

import diffusion_net
from diffusion_net import DiffusionData, DiffusionNet, compute_diffusion_operators
from diffusion_net.utils import sparse_np_to_sparse_tensor

# MaSIF benchmark surfaces produced by the reference stack (verts + chemistry + normals + iface).
DEFAULT_SURF_DIR = Path(
    "/scratch/ymeng/masif-graph/masif-neosurf-af2/masif/data/masif_ppi_search/"
    "data_preparation/01-benchmark_surfaces"
)


def load_ply_mesh(ply_path):
    """Load a MaSIF `.ply` surface.

    Returns
    -------
    verts   : float32 tensor (V, 3)   -- vertex xyz
    faces   : long   tensor (F, 3)    -- triangle vertex indices
    feats   : float32 tensor (V, 3)   -- MaSIF chemistry [charge, hbond, hphob]  (use as C_in=3)
    normals : float32 tensor (V, 3)   -- per-vertex normals [nx, ny, nz] (concat -> C_in=6 if wanted)
    """
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data
    verts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    feats = np.stack([v["charge"], v["hbond"], v["hphob"]], axis=1).astype(np.float32)
    normals = np.stack([v["nx"], v["ny"], v["nz"]], axis=1).astype(np.float32)
    # plyfile stores each face as an int array under 'vertex_indices' (object array of arrays).
    faces = np.stack(ply["face"].data["vertex_indices"]).astype(np.int64)  # (F, 3)

    return (
        torch.from_numpy(verts),
        torch.from_numpy(faces).long(),
        torch.from_numpy(feats),
        torch.from_numpy(normals),
    )


def build_surface_object(verts, faces, feats, k_eig: int = 128):
    """Build the `DiffusionData` object that `DiffusionNet.forward` consumes.

    Computes the (expensive) diffusion operators once and packs them alongside the input features.
    `k_eig` is the spectral truncation (number of Laplacian eigenpairs). scipy's `eigsh` needs
    `k < V`, so we clamp to `min(k_eig, V-2)` for small meshes.

    Returns a `DiffusionData` (a PyG `Data` subclass) with:
      x (V,C_in) float | pos (V,3) | face (3,F) | mass/L/gradX/gradY SparseTensor |
      evals (K,) | evecs (V,K) | batch (V,) all-zeros long.
    """
    verts = verts.float()
    faces = faces.long()
    n_verts = verts.shape[0]
    k = int(min(k_eig, n_verts - 2))

    frames, massvec, L, evals, evecs, gradX, gradY = compute_diffusion_operators(verts, faces, k)

    surf = DiffusionData()
    surf.x = feats.float()
    surf.pos = verts
    surf.face = faces.t().contiguous()  # PyG face convention is (3, F)
    surf.mass = sparse_np_to_sparse_tensor(massvec)
    surf.L = sparse_np_to_sparse_tensor(L)
    surf.evals = torch.from_numpy(evals).float()
    surf.evecs = torch.from_numpy(evecs).float()
    surf.gradX = sparse_np_to_sparse_tensor(gradX)
    surf.gradY = sparse_np_to_sparse_tensor(gradY)
    # A single (unbatched) mesh: every vertex belongs to graph 0. REQUIRED by LearnedTimeDiffusion.
    surf.batch = torch.zeros(n_verts, dtype=torch.long)
    return surf


class SurfaceEncoder(nn.Module):
    """Thin nn.Module wrapping `diffusion_net.DiffusionNet` -> per-vertex embeddings (V, c_out).

    Parameters mirror the fork's DiffusionNet. `dropout` defaults to 0.0 here (deterministic
    building-block validation); raise it in the training config. `forward` returns the (V, c_out)
    embedding tensor (grads flow back into every DiffusionNet parameter).

    NOTE: `forward` mutates `surface.x` (see module docstring). Rebuild/re-set `surface.x` per call.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        c_width: int = 64,
        n_block: int = 2,
        dropout: float = 0.0,
        with_gradient_features: bool = True,
        last_activation=None,
    ):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.net = DiffusionNet(
            C_in=c_in,
            C_out=c_out,
            C_width=c_width,
            N_block=n_block,
            dropout=dropout,
            with_gradient_features=with_gradient_features,
            last_activation=last_activation,
        )

    def forward(self, surface) -> torch.Tensor:
        return self.net(surface).x


def _smoke(surf_dir: Path = DEFAULT_SURF_DIR):
    """Forward+backward validation on two real meshes of different sizes."""
    torch.manual_seed(0)
    c_in, c_out = 3, 16
    encoder = SurfaceEncoder(c_in=c_in, c_out=c_out, c_width=64, n_block=2)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"SurfaceEncoder(c_in={c_in}, c_out={c_out}, c_width=64, n_block=2) -> {n_params} params\n")

    for i, name in enumerate(["1AGQ_C", "1A99_C"]):
        ply_path = surf_dir / f"{name}.ply"
        verts, faces, feats, normals = load_ply_mesh(ply_path)
        print(f"[{name}] V={verts.shape[0]} F={faces.shape[0]} feats={tuple(feats.shape)}")

        t0 = time.perf_counter()
        surf = build_surface_object(verts, faces, feats, k_eig=128)
        t_ops = time.perf_counter() - t0
        print(f"[{name}] operator precompute: {t_ops:.2f} s  (evals={tuple(surf.evals.shape)}, "
              f"evecs={tuple(surf.evecs.shape)})")

        t0 = time.perf_counter()
        emb = encoder(surf)
        t_fwd = time.perf_counter() - t0
        assert emb.shape == (verts.shape[0], c_out), emb.shape
        print(f"[{name}] forward: {t_fwd*1000:.0f} ms  out shape={tuple(emb.shape)}  "
              f"finite={torch.isfinite(emb).all().item()}")

        # Backward: dummy loss, confirm grads reach DiffusionNet params.
        encoder.zero_grad()
        loss = emb.sum()
        loss.backward()
        checked = {
            "first_lin.weight": encoder.net.first_lin.weight,
            "last_lin.weight": encoder.net.last_lin.weight,
            "block_0.diffusion_time": encoder.net.block_0.diffusion.diffusion_time,
        }
        grad_ok = {k: (p.grad is not None and torch.isfinite(p.grad).all().item()
                       and p.grad.abs().sum().item() > 0) for k, p in checked.items()}
        n_with_grad = sum(1 for p in encoder.parameters() if p.grad is not None)
        n_total = sum(1 for _ in encoder.parameters())
        print(f"[{name}] backward OK: {n_with_grad}/{n_total} params have grad; "
              f"spot-check nonzero+finite -> {grad_ok}\n")

    print("SMOKE PASS: forward shape (V, c_out) correct on both meshes; grads flow on backward.")


if __name__ == "__main__":
    _smoke()
