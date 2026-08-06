"""Phase-5 result figures. Generates PNGs into notebooks/figs/ from the gate JSONs.
Colorblind-safe Okabe-Ito palette: learned=blue #0072B2, frozen=orange #E69F00, chance=gray.
Run: python scripts/p5_plots.py"""
import json, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = "/scratch/ymeng/masif-graph"
FIG = f"{ROOT}/notebooks/figs"; os.makedirs(FIG, exist_ok=True)
LEARNED, FROZEN, CHANCE = "#0072B2", "#E69F00", "#9AA0A6"
CELLS = ["HH", "AH", "HA", "AA"]
CELL_LABEL = {"HH": "HH\nholo→holo", "AH": "AH\naf3→holo", "HA": "HA\nholo→af3", "AA": "AA\naf3→af3"}

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 140, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6E8EB", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.edgecolor": "#B0B4B8", "figure.facecolor": "white",
})


def load(patch):
    for stem in ("gate_fullclean","gate_full"):   # prefer the leak-free set
        f=f"{ROOT}/logs/phase5/{stem}_{patch}.json"
        if os.path.exists(f): return json.load(open(f))
    return None


def barlabels(ax, bars, fmt="{:.2f}", dy=0.005):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=8.5, color="#333")


# ---- Fig 1: benchmark funnel (leakage-controlled split) ----
def fig_funnel():
    steps = ["MaSIF-search\ntest list", "sequence-cluster\nclean vs train", "within-test\ndedup",
             "with AF3-apo\n(usable)"]
    vals = [959, 353, 304, 284]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = np.arange(len(vals))[::-1]
    bars = ax.barh(y, vals, color=["#B0B4B8", "#8AB4D8", "#4E8FBF", LEARNED], height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 8, yi, f"{v}", va="center", fontsize=10, color="#333")
    ax.set_yticks(y); ax.set_yticklabels(steps, fontsize=9.5)
    ax.set_xlim(0, 1050); ax.set_xlabel("complexes")
    ax.set_title("Phase-5 eval set: 62% of the nominal test set were train homologs",
                 fontsize=11.5, loc="left", weight="bold")
    ax.annotate("−606 homologs\n(30% seq-id cluster leak)", xy=(353, y[1]), xytext=(560, y[1]+0.35),
                fontsize=8.5, color="#B00", ha="left",
                arrowprops=dict(arrowstyle="->", color="#B00", lw=1.1))
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_funnel.png", bbox_inches="tight"); plt.close(fig)


# ---- Fig 2: 4-cell top-5 recall (learned vs frozen), one panel per patch ----
def fig_cells_top5():
    patches = [("pos_sc", "sc-gated patch"), ("pos", "dense interface patch (deployment-realistic)")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, (pk, title) in zip(axes, patches):
        d = load(pk)
        if d is None:
            ax.set_visible(False); continue
        r = d["results"]; x = np.arange(len(CELLS)); w = 0.38
        fz = [r[f"{c}_frozen"]["top5"] for c in CELLS]
        lr = [r[f"{c}_learned"]["top5"] for c in CELLS]
        b1 = ax.bar(x - w/2, fz, w, color=FROZEN, label="frozen MaSIF")
        b2 = ax.bar(x + w/2, lr, w, color=LEARNED, label="learned (invariant)")
        barlabels(ax, b1); barlabels(ax, b2)
        ch = 5.0 / d["db_chains"]
        ax.axhline(ch, ls=(0, (4, 3)), color=CHANCE, lw=1.2)
        ax.text(3.4, ch + 0.01, "chance", color="#666", fontsize=8, ha="right")
        ax.set_xticks(x); ax.set_xticklabels([CELL_LABEL[c] for c in CELLS], fontsize=9)
        ax.set_title(f"{title}\nDB={d['db_chains']} chains, n={d['n']}", fontsize=10.5, loc="left")
        ax.set_ylim(0, 0.8); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("top-5 recall  (true partner in top 5)")
    axes[0].legend(frameon=False, loc="upper right", fontsize=9.5)
    fig.suptitle("The gate: does the learned encoder retrieve the true binder? (higher = better)",
                 fontsize=12.5, weight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{FIG}/fig2_cells_top5.png", bbox_inches="tight"); plt.close(fig)


# ---- Fig 3: median rank per cell (log), frozen collapse on dense ----
def fig_medrank():
    patches = [("pos_sc", "sc-gated"), ("pos", "dense (deployment)")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, (pk, title) in zip(axes, patches):
        d = load(pk)
        if d is None: ax.set_visible(False); continue
        r = d["results"]; x = np.arange(len(CELLS)); w = 0.38
        fz = [r[f"{c}_frozen"]["median_rank"] for c in CELLS]
        lr = [r[f"{c}_learned"]["median_rank"] for c in CELLS]
        b1 = ax.bar(x - w/2, fz, w, color=FROZEN, label="frozen MaSIF")
        b2 = ax.bar(x + w/2, lr, w, color=LEARNED, label="learned")
        barlabels(ax, b1, "{:.0f}", dy=0.02); barlabels(ax, b2, "{:.0f}", dy=0.02)
        ax.set_yscale("log"); ax.set_ylim(0.8, max(fz) * 2 + 5)
        ax.axhline(d["db_chains"]/2, ls=(0, (4, 3)), color=CHANCE, lw=1.2)
        ax.text(3.4, d["db_chains"]/2*1.05, "chance", color="#666", fontsize=8, ha="right")
        ax.set_xticks(x); ax.set_xticklabels([CELL_LABEL[c] for c in CELLS], fontsize=9)
        ax.set_title(f"{title} patch  (DB={d['db_chains']})", fontsize=10.5, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("median rank of true partner  (log; 1 = best)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9.5)
    fig.suptitle("Frozen MaSIF collapses to ~random on dense interfaces; learned stays at rank 1",
                 fontsize=12.5, weight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{FIG}/fig3_medrank.png", bbox_inches="tight"); plt.close(fig)


# ---- Fig 4: robustness = holo->AF3 degradation (drop from HH) ----
def fig_robustness():
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    groups = []
    for pk, plabel in [("pos_sc", "sc-gated"), ("pos", "dense")]:
        d = load(pk)
        if d is None: continue
        rob = d["robustness"]
        for cell in ["AH", "HA", "AA"]:
            groups.append((f"{plabel}\n{cell}", rob[f"frozen_{cell}_drop"]["top5"], rob[f"learned_{cell}_drop"]["top5"]))
    x = np.arange(len(groups)); w = 0.38
    fz = [g[1] for g in groups]; lr = [g[2] for g in groups]
    b1 = ax.bar(x - w/2, fz, w, color=FROZEN, label="frozen MaSIF")
    b2 = ax.bar(x + w/2, lr, w, color=LEARNED, label="learned")
    barlabels(ax, b1, "{:+.2f}", dy=0.002); barlabels(ax, b2, "{:+.2f}", dy=0.002)
    ax.axhline(0, color="#666", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
    ax.set_ylabel("top-5 recall DROP from holo→holo\n(smaller = more conformation-robust)")
    ax.set_title("Robustness to AI-predicted structures: learned barely degrades, frozen loses a lot",
                 fontsize=11.5, weight="bold", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_robustness.png", bbox_inches="tight"); plt.close(fig)


# ---- Fig 5: retrieval CDF for the AA (fully-predicted) cell, dense patch ----
def fig_cdf():
    d = load("pos")
    if d is None: return
    r = d["results"]; n_db = d["db_chains"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    kmax = 30
    ks = np.arange(1, kmax + 1)
    for key, color, lab, lw in [("AA_learned", LEARNED, "learned  AA (af3→af3)", 2.4),
                                 ("AA_frozen", FROZEN, "frozen  AA (af3→af3)", 2.4),
                                 ("HH_frozen_shuffled", CHANCE, "shuffled control", 1.6)]:
        ranks = np.array(r[key]["ranks"])
        cdf = [(ranks <= k).mean() for k in ks]
        ax.plot(ks, cdf, color=color, lw=lw, label=lab,
                ls="--" if "shuffled" in key else "-")
    ax.set_xlabel("k  (rank cutoff)"); ax.set_ylabel("fraction of queries with true partner in top-k")
    ax.set_xlim(1, kmax); ax.set_ylim(0, 1)
    ax.set_title("Retrieval curve — fully AI-predicted query & database (dense patch)",
                 fontsize=11.5, weight="bold", loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig5_cdf_AA.png", bbox_inches="tight"); plt.close(fig)


# ---- Fig 6 (optional): graph ablation, if no-aa gate exists ----
def fig_ablation():
    def loadn(patch):
        f = f"{ROOT}/logs/phase5/gate_noaaclean_{patch}.json"
        return json.load(open(f)) if os.path.exists(f) else None
    dfull = load("pos"); dnoaa = loadn("pos")
    if dnoaa is None:
        return False
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(CELLS)); w = 0.38
    full = [dfull["results"][f"{c}_learned"]["top5"] for c in CELLS]
    noaa = [dnoaa["results"][f"{c}_learned"]["top5"] for c in CELLS]
    b1 = ax.bar(x - w/2, full, w, color=LEARNED, label="learned  +atom graph")
    b2 = ax.bar(x + w/2, noaa, w, color="#56B4E9", label="learned  no atom graph")
    barlabels(ax, b1); barlabels(ax, b2)
    ax.set_xticks(x); ax.set_xticklabels([CELL_LABEL[c] for c in CELLS], fontsize=9)
    ax.set_ylabel("top-5 recall (dense patch)"); ax.set_ylim(0, 0.8)
    ax.set_title("Graph ablation: does the atom/chem graph earn its keep?", fontsize=11.5, weight="bold", loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=9.5); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig6_ablation.png", bbox_inches="tight"); plt.close(fig)
    return True


if __name__ == "__main__":
    fig_funnel(); print("fig1 funnel")
    fig_cells_top5(); print("fig2 cells top5")
    fig_medrank(); print("fig3 median rank")
    fig_robustness(); print("fig4 robustness")
    fig_cdf(); print("fig5 CDF")
    print("fig6 ablation" if fig_ablation() else "fig6 ablation SKIPPED (no-aa gate not ready)")
    print("figures ->", FIG)
