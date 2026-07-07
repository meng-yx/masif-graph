"""M3 fused encoder: learnable DiffusionNet surface branch ⊕ conformation-invariant chem-graph branch.

The unfreezing lever (Phase-2 lesson #3): the surface descriptor becomes trainable. Architecture:

  per-vertex frozen 80-D MaSIF descriptor  --DiffusionNet-->  per-vertex (surf_width)
       --mean-pool onto surface atoms-->  per-atom surface embedding (surf_width)
  chem graph (covalent connectivity + bond order + rotatability + element chemistry, INVARIANT)
       --RelGraphEncoder-->  per-atom graph embedding (graph_out)
  fuse [surf ⊕ graph] --MLP--> per-atom descriptor (out_dim), L2-normalized.

The SAME encoder is applied to holo and AF3. Straight vs flipped differ only in the DiffusionNet
input (desc_straight / desc_flipped); the graph is identical (chemistry doesn't flip). The graph
uses covalent (invariant) edges only — pose-sensitive spatial edges are dropped (Phase-2 showed they
inject conformation-sensitivity). DiffusionNet operators are conformation-specific but feature-
independent, so they are precomputed/cached per surface (see m3/dataset.py) and only `x` is swapped.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from masif_graph.m3.surface_encoder import SurfaceEncoder
from masif_graph.graph.model import GraphEncoder, AblationConfig


class M3Encoder(nn.Module):
    # node_feat_dim = chem-graph node feats (10 base + 3 element-chem = 13) + 1 flex-depth slot that
    # graph_to_tensors always appends = 14.
    # RESIDUAL design: out_dim == desc_dim (80). The encoder outputs normalize(pooled_frozen_desc +
    # learned_refinement); the refinement's last layer is ZERO-initialized so at init the output IS the
    # frozen mean-pooled descriptor (the proven baseline, af3->holo≈0.82). Training can then only improve
    # on / refine it toward conformation-invariance, rather than relearn the descriptor from scratch.
    def __init__(self, desc_dim=80, node_feat_dim=14, surf_width=64, surf_out=64,
                 graph_dim=64, graph_layers=3, graph_out=32, out_dim=80,
                 use_spatial=False, use_rotatable=True, use_graph=True):
        super().__init__()
        assert out_dim == desc_dim, "residual design needs out_dim == desc_dim"
        self.use_graph = use_graph        # ablation: surface-only vs surface+chem-graph (user's hypothesis)
        self.cfg = AblationConfig(use_covalent=True, use_rotatable=use_rotatable,
                                  use_spatial=use_spatial)
        self.surface_net = SurfaceEncoder(c_in=desc_dim, c_out=surf_out, c_width=surf_width,
                                          n_block=2, dropout=0.0)
        fuse_in = surf_out
        if use_graph:
            self.graph_net = GraphEncoder(self.cfg, dim=graph_dim, n_layers=graph_layers,
                                          out_dim=graph_out)
            self.graph_net.embed = nn.Linear(node_feat_dim, graph_dim)
            fuse_in += graph_out
        last = nn.Linear(128, out_dim)
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)   # refinement = 0 at init -> output = frozen
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, 128), nn.ReLU(), nn.LayerNorm(128), last,
        )
        self.out_dim = out_dim

    def encode_surface(self, surface, desc_input, vertex_surf_idx, n_surf):
        """DiffusionNet on per-vertex desc_input -> pool to surface atoms. Returns (n_surf, surf_out)."""
        surface.x = desc_input                       # (V, desc_dim); forward mutates it
        vemb = self.surface_net(surface)             # (V, surf_out)
        return scatter_mean(vemb, vertex_surf_idx, dim=0, dim_size=n_surf)

    def forward(self, surface, desc_input, vertex_surf_idx, n_surf, graph_tensors, return_reg=False):
        pooled_frozen = scatter_mean(desc_input, vertex_surf_idx, dim=0, dim_size=n_surf)  # (n_surf,80)
        surf_atom = self.encode_surface(surface, desc_input, vertex_surf_idx, n_surf)
        feat = torch.cat([surf_atom, self.graph_net(graph_tensors)], dim=1) if self.use_graph else surf_atom
        refinement = self.fuse(feat)                                                       # 0 at init
        emb = nn.functional.normalize(pooled_frozen + refinement, dim=1)
        if return_reg:
            # magnitude of the deviation from the frozen baseline (anchor toward frozen -> anti-overfit)
            return emb, (refinement ** 2).sum(1).mean()
        return emb
