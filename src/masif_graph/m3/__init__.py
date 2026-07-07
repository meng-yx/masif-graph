"""Phase-3 M3: learnable, conformation-invariant surface-atom encoder.

Architecture (design §M3 ENCODER+GRAPH DECISION):
  - learnable SURFACE encoder (DiffusionNet on the MaSIF mesh) — the *unfreezing* lever;
  - a chemistry-aware, conformation-INVARIANT atom graph (covalent connectivity + bond order +
    rotatability + element/electronegativity/valence) — NOT AtomSurf's distance-only graph;
  - fused, co-trained end-to-end with a contrastive holo<->AF3 conformation-invariance objective.

This is deliberately different from Phase-2 (which fused a chemistry graph with a FROZEN descriptor
at the readout and was NO-GO): here the surface encoder is unfrozen, the objective is invariance, and
pose-sensitive distance edges are dropped.
"""
