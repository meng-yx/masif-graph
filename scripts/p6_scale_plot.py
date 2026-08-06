"""Workstream-B data-scaling curve: Phase-5 gate metric vs training-set size."""
import json, os
import numpy as np, matplotlib.pyplot as plt
ROOT="/scratch/ymeng/masif-graph"; FIG=f"{ROOT}/notebooks/figs"; os.makedirs(FIG,exist_ok=True)
BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#9AA0A6"
plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
                     "grid.color":"#E6E8EB","axes.axisbelow":True,"font.size":11})

def gate(path):
    return json.load(open(path)) if os.path.exists(path) else None
def AA(d):  return d["results"]["AA_learned"]["top5"]
def HH(d):  return d["results"]["HH_learned"]["top5"]
def rob(d): return d["robustness"]["learned_AA_drop"]["top5"]

sizes=[600,1500,3000]
rows=[]
for s in sizes:
    for seed in (0,1):
        d=gate(f"{ROOT}/logs/phase6/gate_scale_{s}_s{seed}_pos.json")
        if d: rows.append((s,seed,AA(d),HH(d),rob(d),d["n"],d["db_chains"]))
full=gate(f"{ROOT}/logs/phase5/gate_fullclean_pos.json")
if full: rows.append((4811,0,AA(full),HH(full),rob(full),full["n"],full["db_chains"]))

print(f"{'size':>6} {'seed':>4} {'AA_top5':>8} {'HH_top5':>8} {'holo->AA drop':>14}  n/DB")
for s,sd,aa,hh,rb,n,db in sorted(rows):
    print(f"{s:>6} {sd:>4} {aa:>8.3f} {hh:>8.3f} {rb:>+14.3f}  {n}/{db}")

# --- figure: AA top5 (deployment metric) vs training-set size, 2 seeds ---
xs=sorted(set(r[0] for r in rows))
fig, ax = plt.subplots(figsize=(7,4.6))
for cell,(get,color,lab) in {"AA":(AA,BLUE,"AA  af3→af3 (deployment)"),
                             "HH":(HH,GREY,"HH  holo→holo")}.items():
    means=[]
    for x in xs:
        ys=[r[2 if cell=="AA" else 3] for r in rows if r[0]==x]
        ax.scatter([x]*len(ys), ys, color=color, s=45, alpha=.7, zorder=3)
        means.append(np.mean(ys))
    ax.plot(xs, means, color=color, lw=2, marker="o", label=lab, zorder=2)
ax.axvline(4811, color="#ccc", ls=":", lw=1)
ax.set_xlabel("training-set size (PPI complexes)"); ax.set_ylabel("learned top-5 retrieval recall")
ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
ax.set_ylim(0, .8); ax.legend(frameon=False, loc="lower right")
ax.set_title("Data-scaling: retrieval quality vs training-set size (leak-free 287-set)",
             weight="bold", fontsize=11.5, loc="left")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_scaling.png", bbox_inches="tight", dpi=140)
print(f"\\nwrote {FIG}/fig_scaling.png")

# verdict heuristic: is AA still climbing from 3000 -> 4811?
try:
    aa3=np.mean([r[2] for r in rows if r[0]==3000]); aa48=[r[2] for r in rows if r[0]==4811][0]
    aa06=np.mean([r[2] for r in rows if r[0]==600])
    print(f"\\nAA top5: 600={aa06:.3f}  3000={aa3:.3f}  4811={aa48:.3f}")
    print(f"slope 3000->4811: {aa48-aa3:+.3f}  |  600->4811: {aa48-aa06:+.3f}")
    print("VERDICT:", "still CLIMBING at 4811 -> more PPI data likely helps (fund larger corpus)"
          if (aa48-aa3) > 0.02 else "≈ SATURATED at 4811 -> scale gives little; invest in the ligand axis")
except Exception as e:
    print("verdict pending:", e)
