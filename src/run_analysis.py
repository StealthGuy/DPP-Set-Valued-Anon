# Code written by claude AI

"""
run_analysis.py — Analisi Sperimentale Completa
SetAnonymizer | He & Naughton, VLDB 2009

Batteria 1 — Replica figure del paper:
  1.1  NCP vs k                     (Fig. 5b)
  1.2  Tempo vs |D|                 (Fig. 6a) — scalabilità lineare
  1.3  NCP vs Fan-out f             (Fig. 7b)
  1.4  Numero partizioni vs k       (originale, complementare)

Batteria 2 — Analisi del generatore sintetico:
  2.1  NCP vs category_bias
  2.2  NCP vs correlation_strength
  2.3  Heatmap NCP(bias × corr)
  2.4  Distribuzione size partizioni per k diversi
  2.5  NCP vs avg_size carrello

Batteria 3 — Trade-off:
  3.1  Curva di Pareto NCP vs k     (privacy vs utility)
"""

import sys
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.hierarchy_loader import HierarchyLoader, build_balanced_hierarchy_for_fanout
from src.utils.dataset_generator import TransactionGenerator
from src.models.transaction import Transaction
from src.models.partition import Partition
from src.core.anonymizer import PartitionAnonymizer
from src.utils.metrics import calculate_database_ncp

# ===========================================================================
# CONFIGURAZIONE
# ===========================================================================

HIERARCHY_PATH = "../data/raw/good_hierarchy.json"
OUTPUT_DIR     = "analysis_results"

# Stile accademico uniforme
STYLE = {
    "figure.figsize":    (7, 4.5),
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "lines.linewidth":   2,
    "lines.markersize":  7,
}
plt.rcParams.update(STYLE)

PALETTE = {
    "primary":   "#2563EB",   # blu acceso
    "secondary": "#16A34A",   # verde
    "accent":    "#DC2626",   # rosso
    "neutral":   "#6B7280",   # grigio
    "light":     "#93C5FD",   # blu chiaro per fill
}


# ===========================================================================
# HELPERS
# ===========================================================================

def load_hierarchy():
    root, leaves = HierarchyLoader.load_from_json(HIERARCHY_PATH)
    return root, leaves


def generate_and_anonymize(root, leaves, count, k,
                            avg_size=5, bias=0.6, corr=0.4):
    """Pipeline completa: genera → anonimizza → ritorna (final_txs, tempo, n_partitions)."""
    domain_size = len(leaves)
    gen = TransactionGenerator(root)
    txs = gen.generate_transactions(count, avg_size=avg_size,
                                    category_bias=bias,
                                    correlation_strength=corr)
    for t in txs:
        t.current_representation = [root]

    anon = PartitionAnonymizer(k=k, root_node=root,
                               total_domain_size=domain_size)
    start = time.time()
    anon.anonymize(Partition(txs, [root]))
    elapsed = time.time() - start

    final_txs = [t for p in anon.final_partitions for t in p.transactions]
    ncp = calculate_database_ncp(final_txs, domain_size)
    n_partitions = len(anon.final_partitions)
    return ncp, elapsed, n_partitions, anon


def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"  ✓  Salvato: {path}")
    plt.close(fig)


# ===========================================================================
# BATTERIA 1 — Replica figure del paper
# ===========================================================================

def plot_1_1_ncp_vs_k():
    """NCP vs k — replica Fig. 5b del paper."""
    print("\n[1/9] NCP vs k ...")
    root, leaves = load_hierarchy()
    k_values = [2, 5, 10, 20, 50]
    ncp_values = []

    for k in k_values:
        ncp, _, _, _ = generate_and_anonymize(root, leaves,
                                               count=800, k=k)
        ncp_values.append(ncp)
        print(f"       k={k:3d}  →  NCP={ncp:.4f}")

    fig, ax = plt.subplots()
    ax.plot(k_values, ncp_values, marker="o",
            color=PALETTE["primary"], label="Partition (local recoding)")
    ax.fill_between(k_values, ncp_values,
                    alpha=0.12, color=PALETTE["light"])
    ax.set_xlabel("Privacy constraint  k")
    ax.set_ylabel("Information Loss  NCP")
    ax.set_title("NCP vs k  —  scalabilità della privacy")
    ax.set_xticks(k_values)
    ax.legend()
    save(fig, "1_1_ncp_vs_k.png")


def plot_1_2_time_vs_size():
    """Tempo vs |D| — replica Fig. 6a del paper (scalabilità lineare)."""
    print("\n[2/9] Tempo vs |D| ...")
    root, leaves = load_hierarchy()
    sizes = [200, 400, 600, 800, 1000, 1500, 2000]
    times = []

    for s in sizes:
        _, elapsed, _, _ = generate_and_anonymize(root, leaves,
                                                   count=s, k=10)
        times.append(elapsed)
        print(f"       |D|={s:5d}  →  {elapsed:.3f}s")

    # Fit lineare per confronto
    coeffs = np.polyfit(sizes, times, 1)
    fit_line = np.poly1d(coeffs)
    xs = np.linspace(min(sizes), max(sizes), 200)

    fig, ax = plt.subplots()
    ax.plot(sizes, times, marker="s",
            color=PALETTE["primary"], label="Tempo misurato")
    ax.plot(xs, fit_line(xs), "--",
            color=PALETTE["accent"], alpha=0.7, label="Fit lineare O(N)")
    ax.set_xlabel("Numero di transazioni  |D|")
    ax.set_ylabel("Tempo di esecuzione  (s)")
    ax.set_title("Scalabilità: Tempo vs |D|  —  crescita lineare")
    ax.legend()
    save(fig, "1_2_time_vs_size.png")


def plot_1_3_ncp_vs_fanout():
    """NCP vs fan-out f — replica Fig. 7b del paper."""
    print("\n[3/9] NCP vs fan-out ...")
    fanouts = [2, 3, 5, 7, 10, 15, 20]
    ncp_values = []
    N_ITEMS = 200

    for f in fanouts:
        root, leaves = build_balanced_hierarchy_for_fanout(f, N_ITEMS)
        domain_size = len(leaves)
        gen = TransactionGenerator(root)
        txs = gen.generate_transactions(600, avg_size=5,
                                        category_bias=0.6,
                                        correlation_strength=0.4)
        for t in txs:
            t.current_representation = [root]

        anon = PartitionAnonymizer(k=10, root_node=root,
                                   total_domain_size=domain_size)
        anon.anonymize(Partition(txs, [root]))
        final_txs = [t for p in anon.final_partitions for t in p.transactions]
        ncp = calculate_database_ncp(final_txs, domain_size)
        ncp_values.append(ncp)
        print(f"       f={f:3d}  →  NCP={ncp:.4f}")

    fig, ax = plt.subplots()
    bars = ax.bar([str(f) for f in fanouts], ncp_values,
                  color=PALETTE["primary"], alpha=0.75, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    ax.set_xlabel("Fan-out  f  della gerarchia")
    ax.set_ylabel("Information Loss  NCP")
    ax.set_title("Sensibilità al Fan-out: NCP vs f")
    save(fig, "1_3_ncp_vs_fanout.png")


def plot_1_4_partitions_vs_k():
    """Numero di partizioni finali vs k — originale, complementare al paper."""
    print("\n[4/9] Partizioni vs k ...")
    root, leaves = load_hierarchy()
    k_values = [2, 5, 10, 20, 50, 100]
    n_parts = []
    ncp_vals = []

    for k in k_values:
        ncp, _, n_p, _ = generate_and_anonymize(root, leaves,
                                                 count=800, k=k)
        n_parts.append(n_p)
        ncp_vals.append(ncp)
        print(f"       k={k:4d}  →  partizioni={n_p}  NCP={ncp:.4f}")

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(k_values, n_parts, marker="o",
             color=PALETTE["primary"], label="# Partizioni")
    ax2.plot(k_values, ncp_vals, marker="^", linestyle="--",
             color=PALETTE["accent"], label="NCP")

    ax1.set_xlabel("Privacy constraint  k")
    ax1.set_ylabel("Numero di partizioni finali", color=PALETTE["primary"])
    ax2.set_ylabel("Information Loss  NCP", color=PALETTE["accent"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["primary"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["accent"])
    ax1.set_title("Granularità vs Privacy: Partizioni e NCP al variare di k")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    save(fig, "1_4_partitions_vs_k.png")


# ===========================================================================
# BATTERIA 2 — Analisi del generatore sintetico
# ===========================================================================

def plot_2_1_ncp_vs_bias():
    """NCP vs category_bias — effetto del clustering sui dati."""
    print("\n[5/9] NCP vs category_bias ...")
    root, leaves = load_hierarchy()
    bias_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = {k: [] for k in [5, 20]}

    for k in results:
        for b in bias_values:
            ncp, _, _, _ = generate_and_anonymize(root, leaves,
                                                   count=800, k=k,
                                                   bias=b, corr=0.4)
            results[k].append(ncp)
            print(f"       k={k}  bias={b:.1f}  →  NCP={ncp:.4f}")

    fig, ax = plt.subplots()
    ax.plot(bias_values, results[5], marker="o",
            color=PALETTE["primary"], label="k=5")
    ax.plot(bias_values, results[20], marker="s", linestyle="--",
            color=PALETTE["secondary"], label="k=20")
    ax.set_xlabel("Category Bias")
    ax.set_ylabel("Information Loss  NCP")
    ax.set_title("NCP vs Category Bias\n(bias alto → dati più clusterizzati → meno generalizzazione)")
    ax.legend()
    save(fig, "2_1_ncp_vs_bias.png")


def plot_2_2_ncp_vs_corr():
    """NCP vs correlation_strength — effetto della correlazione locale."""
    print("\n[6/9] NCP vs correlation_strength ...")
    root, leaves = load_hierarchy()
    corr_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = {k: [] for k in [5, 20]}

    for k in results:
        for c in corr_values:
            ncp, _, _, _ = generate_and_anonymize(root, leaves,
                                                   count=800, k=k,
                                                   bias=0.6, corr=c)
            results[k].append(ncp)
            print(f"       k={k}  corr={c:.1f}  →  NCP={ncp:.4f}")

    fig, ax = plt.subplots()
    ax.plot(corr_values, results[5], marker="o",
            color=PALETTE["primary"], label="k=5")
    ax.plot(corr_values, results[20], marker="s", linestyle="--",
            color=PALETTE["secondary"], label="k=20")
    ax.set_xlabel("Correlation Strength")
    ax.set_ylabel("Information Loss  NCP")
    ax.set_title("NCP vs Correlation Strength\n(correlazione locale tra item sibling)")
    ax.legend()
    save(fig, "2_2_ncp_vs_corr.png")


def plot_2_3_heatmap_bias_corr():
    """Heatmap NCP(bias × corr) — il grafico più pubblicabile della batteria."""
    print("\n[7/9] Heatmap NCP(bias × corr) ...")
    root, leaves = load_hierarchy()
    bias_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    corr_values = [0.0, 0.2, 0.4, 0.6, 0.8]
    matrix = np.zeros((len(corr_values), len(bias_values)))

    for i, c in enumerate(corr_values):
        for j, b in enumerate(bias_values):
            ncp, _, _, _ = generate_and_anonymize(root, leaves,
                                                   count=600, k=10,
                                                   bias=b, corr=c)
            matrix[i, j] = ncp
            print(f"       bias={b:.1f}  corr={c:.1f}  →  NCP={ncp:.4f}")

    # Colormap: verde (basso NCP = buono) → rosso (alto NCP = cattivo)
    cmap = LinearSegmentedColormap.from_list(
        "ncp_cmap", ["#16A34A", "#FACC15", "#DC2626"])

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                   vmin=matrix.min(), vmax=matrix.max())
    plt.colorbar(im, ax=ax, label="Information Loss  NCP")

    ax.set_xticks(range(len(bias_values)))
    ax.set_yticks(range(len(corr_values)))
    ax.set_xticklabels([f"{b:.1f}" for b in bias_values])
    ax.set_yticklabels([f"{c:.1f}" for c in corr_values])
    ax.set_xlabel("Category Bias")
    ax.set_ylabel("Correlation Strength")
    ax.set_title("Heatmap: NCP in funzione di Bias e Correlazione  (k=10)")

    # Annotazioni numeriche nelle celle
    for i in range(len(corr_values)):
        for j in range(len(bias_values)):
            ax.text(j, i, f"{matrix[i,j]:.3f}",
                    ha="center", va="center",
                    fontsize=8, color="white",
                    fontweight="bold")
    save(fig, "2_3_heatmap_bias_corr.png")


def plot_2_4_partition_size_distribution():
    """Distribuzione della dimensione delle partizioni per k diversi."""
    print("\n[8/9] Distribuzione size partizioni ...")
    root, leaves = load_hierarchy()
    k_configs = [
        (2,  PALETTE["primary"],   "k=2"),
        (10, PALETTE["secondary"], "k=10"),
        (50, PALETTE["accent"],    "k=50"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, (k, color, label) in zip(axes, k_configs):
        _, _, _, anon = generate_and_anonymize(root, leaves,
                                               count=1000, k=k)
        sizes = [len(p) for p in anon.final_partitions]
        print(f"       k={k:3d}  →  partizioni={len(sizes)}  "
              f"media={np.mean(sizes):.1f}  max={max(sizes)}")

        ax.hist(sizes, bins=20, color=color, alpha=0.75,
                edgecolor="white")
        ax.axvline(np.mean(sizes), color="black", linestyle="--",
                   linewidth=1.2, label=f"media={np.mean(sizes):.1f}")
        ax.set_title(label)
        ax.set_xlabel("Dimensione partizione (tx)")
        ax.set_ylabel("Frequenza")
        ax.legend(fontsize=8)

    fig.suptitle("Distribuzione della Dimensione delle Partizioni",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "2_4_partition_size_distribution.png")


def plot_2_5_ncp_vs_avg_size():
    """NCP vs avg_size del carrello — ipotesi: più item → più collisioni → NCP minore."""
    print("\n[9/9] NCP vs avg_size ...")
    root, leaves = load_hierarchy()
    sizes = [2, 3, 5, 7, 10]
    results = {k: [] for k in [5, 20]}

    for k in results:
        for s in sizes:
            ncp, _, _, _ = generate_and_anonymize(root, leaves,
                                                   count=800, k=k,
                                                   avg_size=s,
                                                   bias=0.6, corr=0.4)
            results[k].append(ncp)
            print(f"       k={k}  avg_size={s}  →  NCP={ncp:.4f}")

    fig, ax = plt.subplots()
    ax.plot(sizes, results[5], marker="o",
            color=PALETTE["primary"], label="k=5")
    ax.plot(sizes, results[20], marker="s", linestyle="--",
            color=PALETTE["secondary"], label="k=20")
    ax.set_xlabel("Dimensione media del carrello  avg_size")
    ax.set_ylabel("Information Loss  NCP")
    ax.set_title("NCP vs Dimensione Media Transazione\n"
                 "(carrelli più grandi → più collisioni → meno generalizzazione attesa)")
    ax.legend()
    save(fig, "2_5_ncp_vs_avg_size.png")


# ===========================================================================
# BATTERIA 3 — Trade-off Privacy / Utility
# ===========================================================================

def plot_3_1_pareto_curve():
    """
    Curva di Pareto NCP vs k.
    Mostra il trade-off fondamentale: più privacy → più information loss.
    L'area sotto la curva è il costo totale dell'anonimizzazione.
    """
    print("\n[BONUS] Curva di Pareto NCP vs k ...")
    root, leaves = load_hierarchy()
    k_values = [2, 5, 10, 20, 30, 50, 75, 100]
    ncp_values = []

    for k in k_values:
        ncp, _, _, _ = generate_and_anonymize(root, leaves,
                                               count=1000, k=k)
        ncp_values.append(ncp)
        print(f"       k={k:4d}  →  NCP={ncp:.4f}")

    fig, ax = plt.subplots()

    # Area sotto la curva = costo totale
    ax.fill_between(k_values, ncp_values,
                    alpha=0.15, color=PALETTE["primary"],
                    label="Costo totale anonimizzazione")
    ax.plot(k_values, ncp_values, marker="o",
            color=PALETTE["primary"], label="NCP (Partition algorithm)")

    # Annotazione zona "ottimale"
    idx_opt = int(len(k_values) * 0.3)
    ax.annotate("Zona ottimale\nprivacy/utility",
                xy=(k_values[idx_opt], ncp_values[idx_opt]),
                xytext=(k_values[idx_opt] + 8, ncp_values[idx_opt] - 0.03),
                arrowprops=dict(arrowstyle="->", color=PALETTE["accent"]),
                fontsize=9, color=PALETTE["accent"])

    ax.set_xlabel("Privacy constraint  k  →  più privacy")
    ax.set_ylabel("Information Loss  NCP  →  meno utilità")
    ax.set_title("Trade-off Privacy / Utility  —  Curva di Pareto")
    ax.legend()

    # Secondo asse X per leggibilità
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(k_values)
    ax2.set_xticklabels([f"k={v}" for v in k_values], fontsize=7)

    save(fig, "3_1_pareto_privacy_utility.png")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("=" * 60)
    print("  ANALISI SPERIMENTALE COMPLETA — SetAnonymizer")
    print("  He & Naughton, VLDB 2009")
    print("=" * 60)
    t_total = time.time()

    # Batteria 1 — Replica paper
    print("\n━━━  BATTERIA 1: Replica figure del paper  ━━━")
    plot_1_1_ncp_vs_k()
    plot_1_2_time_vs_size()
    plot_1_3_ncp_vs_fanout()
    plot_1_4_partitions_vs_k()

    # Batteria 2 — Generatore sintetico
    print("\n━━━  BATTERIA 2: Analisi generatore sintetico  ━━━")
    plot_2_1_ncp_vs_bias()
    plot_2_2_ncp_vs_corr()
    plot_2_3_heatmap_bias_corr()
    plot_2_4_partition_size_distribution()
    plot_2_5_ncp_vs_avg_size()

    # Batteria 3 — Trade-off
    print("\n━━━  BATTERIA 3: Trade-off Privacy/Utility  ━━━")
    plot_3_1_pareto_curve()

    elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"  ✓  Analisi completata in {elapsed:.1f}s")
    print(f"  ✓  10 grafici salvati in: {OUTPUT_DIR}/")
    print(f"{'=' * 60}")