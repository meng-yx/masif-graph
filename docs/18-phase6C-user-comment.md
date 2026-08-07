# Phase-6 Workstream C — async steering channel

This is how you (the user) steer the headless agent **without interrupting it**. Add a comment
beginning `### 🧑 USER:` under any step header below (or at the end) at any time. The agent
re-reads this file at every step boundary, replies inline with `### 🤖 AGENT:`, acts on it, and
keeps going. A comment is **steering, not a stop signal**. The agent never edits your lines.

The agent mirrors each running-log step header here as it starts. Budget checkpoints (if a
planned action would cross a budget gate) also appear here.

---

## Mission (locked in PHASE6C_HANDOFF.md)
Build the ligand-capable unified 26-D retrieval encoder (Path B: ligand atoms as graph nodes) on
the combined PPI(≥3k) + PDBbind-refined(~5.3k) corpus; eval = do-no-harm PPI gate + mixed
held-out + neosurface (MolGlueDB) benchmark. Conductor, CHF 100 budget. Done = `logs/PHASE6C_DONE`.

---

## Step log (mirrored from docs/progress/phase6C-log.md)

<!-- agent appends "### <step>" headers here as it starts each step -->
