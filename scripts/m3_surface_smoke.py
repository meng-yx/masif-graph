#!/usr/bin/env python
"""Smoke test for the M3 DiffusionNet surface encoder building block.

Run with the atomsurf env python (has diffusion_net + torch):
  export PYTHONPATH=/scratch/ymeng/masif-graph/src
  /work/upthomae/Meng/conda_envs/atomsurf_h100/bin/python scripts/m3_surface_smoke.py

Loads two real MaSIF meshes of different sizes, builds the DiffusionData surface object,
runs a forward (expects (V, 16)) and a backward (confirms grads flow to DiffusionNet params).
"""
import sys

sys.path.insert(0, "/scratch/ymeng/masif-graph/src")

from masif_graph.m3.surface_encoder import _smoke

if __name__ == "__main__":
    _smoke()
