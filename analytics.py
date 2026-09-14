"""
analytics.py
------------
Generate a 4-panel analytics figure comparing ORB's L1 estimator
against ordinary least squares, demonstrating robustness to corruption
and sink (leak/consumption) detection.

Usage:  python3 analytics.py          → saves output/analytics.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from orb.compile   import compile as compile_graph
from orb.run_graph import _l1_solve
from orb.decode    import decode
from orb.demo      import run_demo
from orb.water_world import WATER_COLUMN_MAPPING


# ── Style ──────────────────────────────────────────────────────────────────────
BG       = "#07090d"
SURFACE  = "#0c1018"
BORDER   = "#1d2a3a"
TEXT     = "#dde3ec"
TEXT2    = "#7a8a9a"
BLUE     = "#4f9cf9"
TEAL     = "#2ec4a9"
AMBER    = "#e8952a"
RED      = "#e84040"
GREEN    = "#30c97e"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   SURFACE,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT,
    "text.color":       TEXT,
    "xtick.color":      TEXT2,
    "ytick.color":      TEXT2,
    "grid.color":       BORDER,
    "grid.alpha":       0.5,
    "font.family":      "sans-serif",
    "font.size":        9,
})


# ── Least-squares solver (for comparison) ──────────────────────────────────────

def _ls_solve(compiled: dict) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least-squares with hard balance constraints (via null-space)."""
    H     = compiled["H"]
    y     = compiled["y"]
    w     = compiled["w"]
    A_eq  = compiled["A_eq"]
    b_eq  = compiled["b_eq"]
    n     = H.shape[1]

    # Particular solution satisfying A_eq @ x = b_eq
    x_part, _, _, _ = np.linalg.lstsq(A_eq, b_eq, rcond=None)

    # Null space of A_eq
    _, s, Vt = np.linalg.svd(A_eq, full_matrices=True)
    rank_A = int(np.sum(s > 1e-10))
    Z = Vt[rank_A:].T   # columns span null(A_eq)

    if Z.shape[1] == 0:
        x_hat = x_part
    else:
        # Minimize || W^{1/2} (H @ (x_part + Z @ alpha) - y) ||_2
        W_sqrt = np.diag(np.sqrt(w))
        HZ = W_sqrt @ H @ Z
        rhs = W_sqrt @ (y - H @ x_part)
        alpha, _, _, _ = np.linalg.lstsq(HZ, rhs, rcond=None)
        x_hat = x_part + Z @ alpha

    residuals = y - H @ x_hat
    return x_hat, residuals


# ── Run both estimators on fuel CSV ────────────────────────────────────────────

FUEL_MAPPING = {
    "entity_column": "sensor_id", "value_column": "value",
    "time_column": "timestamp", "channel_column": "channel",
    "channel_map": {"opening": "node", "flow": "edge", "eod": "node", "consumption": "sink"},
    "from_column": "from_node", "to_column": "to_node",
    "id_pattern": None, "excluded_channels": [],
    "notes": "Fuel supply chain with known consumption burn rates.",
}

FUEL_TRUE = {
    "PORT_HAVEN": 11000, "FOB_IRONSIDE": 6300, "FOB_KESTREL": 4600,
    "OP_TALON": 1750, "OP_VIPER": 930,
}

import contextlib, io

with contextlib.redirect_stdout(io.StringIO()):
    fuel_result = run_demo("tests/fixtures/fuel_E_consumption.csv", column_mapping=FUEL_MAPPING)
    water_result = run_demo(
        "tests/fixtures/demo_water.csv", truth_path="tests/fixtures/demo_water_truth.json",
        sinks="unknown", column_mapping=WATER_COLUMN_MAPPING,
    )

fuel_compiled = fuel_result["compiled"]
fuel_graph    = fuel_result["graph"]

# L1
l1_x, l1_res = _l1_solve(fuel_compiled)
l1_decoded    = decode(l1_x, l1_res, fuel_compiled, fuel_graph)

# LS
ls_x, ls_res = _ls_solve(fuel_compiled)
ls_decoded    = decode(ls_x, ls_res, fuel_compiled, fuel_graph)

node_ids = [n["id"] for n in l1_decoded["nodes"]]
l1_qty   = {n["id"]: n["qty"] for n in l1_decoded["nodes"]}
ls_qty   = {n["id"]: n["qty"] for n in ls_decoded["nodes"]}


# ── Build the figure ───────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 9))
fig.suptitle("ORB  —  L1 Robust Estimation vs Least Squares",
             fontsize=16, fontweight="bold", color=TEXT, y=0.97)
fig.text(0.5, 0.935,
         "Corruption: OP_VIPER end-of-day reported 980 (true 930, +50 error)",
         ha="center", fontsize=10, color=TEXT2)

gs = GridSpec(2, 2, hspace=0.38, wspace=0.3,
              left=0.08, right=0.95, top=0.90, bottom=0.08)


# ── Panel 1: Node quantities — true vs L1 vs LS ──────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
x_pos = np.arange(len(node_ids))
bar_w = 0.25

true_vals = [FUEL_TRUE[n] for n in node_ids]
l1_vals   = [l1_qty[n] for n in node_ids]
ls_vals   = [ls_qty[n] for n in node_ids]

ax1.bar(x_pos - bar_w, true_vals, bar_w, label="True",          color=TEAL,  alpha=0.85)
ax1.bar(x_pos,         l1_vals,   bar_w, label="L1 (ORB)",      color=BLUE,  alpha=0.85)
ax1.bar(x_pos + bar_w, ls_vals,   bar_w, label="Least Squares", color=AMBER, alpha=0.85)

ax1.set_xticks(x_pos)
short_names = [n.replace("FOB_", "").replace("OP_", "").replace("PORT_", "") for n in node_ids]
ax1.set_xticklabels(short_names, fontsize=8)
ax1.set_ylabel("End-of-Day Quantity")
ax1.set_title("Decoded Node Quantities", fontsize=11, fontweight="600", color=TEXT)
ax1.legend(fontsize=8, loc="upper right", framealpha=0.3, edgecolor=BORDER)
ax1.grid(axis="y", linestyle="--", alpha=0.3)

# Highlight OP_VIPER
viper_idx = node_ids.index("OP_VIPER")
ax1.annotate("corrupted\nclaim",
             xy=(viper_idx + bar_w, ls_qty["OP_VIPER"]),
             xytext=(viper_idx + 0.8, ls_qty["OP_VIPER"] + 400),
             fontsize=7, color=RED, ha="center",
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))


# ── Panel 2: Estimation error per node ────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])

l1_err = [abs(l1_qty[n] - FUEL_TRUE[n]) for n in node_ids]
ls_err = [abs(ls_qty[n] - FUEL_TRUE[n]) for n in node_ids]

ax2.bar(x_pos - 0.15, l1_err, 0.3, label="L1 (ORB)",      color=BLUE,  alpha=0.85)
ax2.bar(x_pos + 0.15, ls_err, 0.3, label="Least Squares", color=AMBER, alpha=0.85)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(short_names, fontsize=8)
ax2.set_ylabel("Absolute Error")
ax2.set_title("Estimation Error by Node", fontsize=11, fontweight="600", color=TEXT)
ax2.legend(fontsize=8, loc="upper right", framealpha=0.3, edgecolor=BORDER)
ax2.grid(axis="y", linestyle="--", alpha=0.3)

# Summary stats
l1_mae  = np.mean(l1_err)
ls_mae  = np.mean(ls_err)
ax2.text(0.02, 0.95, f"L1 MAE: {l1_mae:.1f}", transform=ax2.transAxes,
         fontsize=9, fontweight="600", color=GREEN, va="top")
ax2.text(0.02, 0.86, f"LS MAE: {ls_mae:.1f}", transform=ax2.transAxes,
         fontsize=9, fontweight="600", color=RED, va="top")


# ── Panel 3: Residuals — L1 sparse vs LS spread ──────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])

claim_labels = [c["id"] for c in fuel_graph["claims"]]
n_claims = len(claim_labels)
x_c = np.arange(n_claims)

ax3.bar(x_c - 0.15, np.abs(l1_res), 0.3, label="L1 (ORB)",      color=BLUE,  alpha=0.85)
ax3.bar(x_c + 0.15, np.abs(ls_res), 0.3, label="Least Squares", color=AMBER, alpha=0.85)

# Mark the corrupted claim
corrupted_idx = None
for i, c in enumerate(fuel_graph["claims"]):
    if c.get("ref") == "OP_VIPER" and c["type"] == "node":
        corrupted_idx = i
        break

if corrupted_idx is not None:
    ax3.annotate("L1 isolates\ncorruption here",
                 xy=(corrupted_idx - 0.15, abs(l1_res[corrupted_idx])),
                 xytext=(corrupted_idx - 2.5, abs(l1_res[corrupted_idx]) + 8),
                 fontsize=7, color=GREEN, ha="center",
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))

ax3.set_xticks(x_c)
# Label claims by their ref
ref_labels = []
for c in fuel_graph["claims"]:
    ref = c.get("ref", "?")
    if c["type"] == "edge":
        # Find the edge display
        edge = next((e for e in fuel_graph["edges"] if e["id"] == ref), None)
        if edge:
            ref = f"{edge['from'][:4]}→{edge['to'][:4]}"
    else:
        ref = ref.replace("FOB_", "").replace("OP_", "").replace("PORT_", "")
    ref_labels.append(ref)

ax3.set_xticklabels(ref_labels, fontsize=6.5, rotation=35, ha="right")
ax3.set_ylabel("|Residual|")
ax3.set_title("Claim Residuals: L1 is Sparse, LS Spreads Error",
              fontsize=11, fontweight="600", color=TEXT)
ax3.legend(fontsize=8, loc="upper left", framealpha=0.3, edgecolor=BORDER)
ax3.grid(axis="y", linestyle="--", alpha=0.3)


# ── Panel 4: Sink detection (water network) ──────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])

water_decoded = water_result["decoded"]
water_sinks   = sorted(water_decoded["sinks"], key=lambda s: abs(s["sink"]), reverse=True)

sink_ids  = [s["id"] for s in water_sinks]
sink_vals = [abs(s["sink"]) for s in water_sinks]
colors    = [RED if s["id"] == "J_03" else BLUE for s in water_sinks]

bars = ax4.barh(range(len(sink_ids)), sink_vals, color=colors, alpha=0.85, height=0.6)
ax4.set_yticks(range(len(sink_ids)))
ax4.set_yticklabels(sink_ids, fontsize=9)
ax4.invert_yaxis()
ax4.set_xlabel("|Sink Magnitude|")
ax4.set_title("Water Network: Leak Detection via Unknown Sinks",
              fontsize=11, fontweight="600", color=TEXT)
ax4.grid(axis="x", linestyle="--", alpha=0.3)

# Label the true leak
leak_idx = sink_ids.index("J_03")
ax4.annotate(f"  True leak (J_03)\n  |sink| = {sink_vals[leak_idx]:.0f}",
             xy=(sink_vals[leak_idx], leak_idx),
             xytext=(sink_vals[leak_idx] * 0.4, leak_idx + 0.6),
             fontsize=8, fontweight="600", color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

# Correctable k badge
k = water_result["report"]["correctable_k"]
ax4.text(0.98, 0.95, f"correctable_k = {k}",
         transform=ax4.transAxes, fontsize=9, fontweight="600",
         color=GREEN, ha="right", va="top",
         bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE, edgecolor=GREEN, alpha=0.8))


# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)
out_path = "output/analytics.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"\nSaved → {out_path}")
plt.close(fig)
