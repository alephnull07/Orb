#!/usr/bin/env python3
"""
scripts/benchmark_estimators.py
-------------------------------
Least squares vs ORB's weighted L1 vs brute force, on the same compiled
system for ten fixture scenarios with known ground truth.

  least squares  min Σ wᵢ (yᵢ − Hᵢx)²  s.t. A x = b          (closed form, KKT)
  ORB (L1)       min Σ wᵢ |yᵢ − Hᵢx| + λ Σ s  s.t. A x = b    (LP)
  brute force    smallest set S of rows to discard such that the remaining rows
                 can be fit EXACTLY under A x = b (the L0 / minimum-support
                 decoder), searched over every subset up to a budget.

Every method sees exactly the H, y, w, A_eq, b_eq that the pipeline compiled
for that scenario.  Flags are |residual| > 5 (the pipeline threshold); for
brute force the flags are the discarded set.

    python3 scripts/benchmark_estimators.py            # run + write docs/benchmark.json
    python3 scripts/benchmark_estimators.py --chart    # also render docs/benchmark.png
"""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from orb.presets import run_preset                          # noqa: E402
from orb.run_graph import _l1_solve                          # noqa: E402
from verify_fixtures import TRUTH, WATER_A_TRUE              # noqa: E402

THRESH = 5.0
BRUTE_BUDGET = 60_000        # LP evaluations per scenario before giving up
BRUTE_MAX_K = 6

FUEL_TRUE = TRUTH["fuel_C_clean"]

# (preset id, short label, node truth or None=clean/self-consistent, planted corrupt rows)
# planted rows are matched on (type, ref-or-"from->to", value)
SCENARIOS = [
    ("fuel-two-false-reports", "Fuel: 2 independent lies", FUEL_TRUE,
     [("node", "OP_TALON", 1450), ("edge", "FOB_IRONSIDE->OP_TALON", 1300)]),
    ("fuel-one-bad-reporter", "Fuel: 1 correlated reporter", FUEL_TRUE,
     [("node", "FOB_KESTREL", 5080), ("edge", "FOB_KESTREL->OP_TALON", 480), ("edge", "FOB_KESTREL->OP_VIPER", 580)]),
    ("fuel-consumption", "Fuel: known burn + 1 lie", TRUTH["fuel_E_consumption"],
     [("node", "OP_VIPER", 980)]),
    ("fuel-past-guarantee", "Fuel: 4 lies (past k=1)", FUEL_TRUE,
     [("node", "OP_TALON", 1450), ("node", "OP_VIPER", 1400), ("node", "FOB_KESTREL", 4850),
      ("edge", "FOB_IRONSIDE->OP_TALON", 1300)]),
    ("supply-three-formats", "Supply: 3 formats, 1 lie ×3", TRUTH["supply"],
     [("node", "OP_DELTA", 520)]),
    ("medical-four-sources", "Medical: 4 sources, 2 lies", TRUTH["medical"],
     [("node", "AID_STN_2", 620), ("edge", "FWD_HOSP_NORTH->AID_STN_3", 340)]),
    ("water-find-the-loss", "Water: leak, no lies", WATER_A_TRUE, []),
    ("water-lie-or-leak", "Water: leak + 1 lie", WATER_A_TRUE,
     [("node", "ZONE_1", 1500)]),
    ("large-network", "15-node network, clean", None, []),
    ("fuel-audit-meters", "Fuel: metered sinks + 1 lie", TRUTH["fuel_E_consumption"],
     [("node", "OP_VIPER", 980)]),
]


# ── solvers ──────────────────────────────────────────────────────────────────

def _sink_cols(compiled):
    return sorted(c for k, c in compiled["index"].items() if k.startswith("sink_"))


def solve_ls(compiled, rows=None):
    """Weighted least squares with hard equality constraints (KKT system).
    Sink columns are unbounded here — plain least squares has no sign logic."""
    H, y, w, A, b = compiled["H"], compiled["y"], compiled["w"], compiled["A_eq"], compiled["b_eq"]
    if rows is not None:
        H, y, w = H[rows], y[rows], w[rows]
    n, p = H.shape[1], A.shape[0]
    Wh = H * w[:, None]
    K = np.block([[2 * H.T @ Wh, A.T], [A, np.zeros((p, p))]])
    rhs = np.concatenate([2 * H.T @ (w * y), b])
    sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
    x = sol[:n]
    return x, compiled["y"] - compiled["H"] @ x


def solve_l1(compiled):
    return _l1_solve(compiled)


def _exact_fit_lp(compiled, keep):
    """Feasibility LP: fit kept rows exactly under the balance, sinks ≥ 0,
    minimising the sink penalty.  Returns x or None if infeasible."""
    H, y, A, b, w_sink = compiled["H"], compiled["y"], compiled["A_eq"], compiled["b_eq"], compiled["w_sink"]
    n = H.shape[1]
    sinks = set(_sink_cols(compiled))
    A_full = np.vstack([A, H[keep]])
    b_full = np.concatenate([b, y[keep]])
    bounds = [(0.0, None) if c in sinks else (None, None) for c in range(n)]
    c = w_sink if w_sink.any() else np.zeros(n)
    res = linprog(c, A_eq=A_full, b_eq=b_full, bounds=bounds, method="highs")
    return res.x if res.status == 0 else None


def solve_brute(compiled):
    """Minimum-support decoder: smallest S such that the rows outside S fit
    exactly.  Exhaustive over subsets, budgeted.  Returns (x, residuals, S, evals, finished)."""
    m = compiled["H"].shape[0]
    evals = 0
    for k in range(0, min(BRUTE_MAX_K, m) + 1):
        best = None
        for S in itertools.combinations(range(m), k):
            evals += 1
            if evals > BRUTE_BUDGET:
                return None, None, None, evals, False
            keep = [i for i in range(m) if i not in S]
            x = _exact_fit_lp(compiled, keep)
            if x is not None:
                obj = float(compiled["w_sink"] @ x)
                if best is None or obj < best[0]:
                    best = (obj, x, set(S))
        if best is not None:
            x = best[1]
            return x, compiled["y"] - compiled["H"] @ x, best[2], evals, True
    return None, None, None, evals, False


# ── scoring ──────────────────────────────────────────────────────────────────

def node_error(compiled, graph, x, truth):
    idx = compiled["index"]
    errs = []
    for nid, tv in truth.items():
        key = f"qty_{nid}"
        if key in idx:
            errs.append(abs(float(x[idx[key]]) - tv))
    return max(errs) if errs else float("nan")


def planted_rows(graph, planted):
    edge_name = {e["id"]: f"{e['from']}->{e['to']}" for e in graph["edges"]}
    out = set()
    for i, c in enumerate(graph["claims"]):
        ref = edge_name.get(c.get("ref"), c.get("ref"))
        for (t, r, v) in planted:
            if c["type"] == t and ref == r and abs(float(c["value"]) - v) < 1e-6:
                out.add(i)
    return out


def prf(flagged: set, truth_rows: set):
    tp = len(flagged & truth_rows)
    fp = len(flagged - truth_rows)
    fn = len(truth_rows - flagged)
    if not truth_rows and not flagged:
        return 1.0, 1.0, 1.0, fp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, fp


def run_all():
    rows = []
    for pid, label, truth, planted in SCENARIOS:
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_preset(pid)
        comp, graph = res["compiled"], res["graph"]
        m, n = comp["H"].shape
        corrupt = planted_rows(graph, planted)
        if planted:
            assert corrupt, f"{pid}: planted rows not found in claims"

        # a clean scenario: truth is the unique consistent solution
        if truth is None:
            x0, _ = solve_ls(comp)
            truth = {nid: float(x0[c]) for k, c in comp["index"].items() if k.startswith("qty_") for nid in [k[4:]]}

        entry = {"id": pid, "label": label, "m_claims": m, "n_vars": n,
                 "k": res["report"]["correctable_k"], "n_planted": len(corrupt), "methods": {}}

        # least squares
        t = time.perf_counter(); x, r = solve_ls(comp); dt = time.perf_counter() - t
        fl = {i for i in range(m) if abs(r[i]) > THRESH}
        p_, r_, f1, fp = prf(fl, corrupt)
        entry["methods"]["least_squares"] = dict(max_node_error=node_error(comp, graph, x, truth),
                                                 precision=p_, recall=r_, f1=f1, false_flags=fp,
                                                 n_flagged=len(fl), seconds=dt, finished=True)
        # ORB L1
        t = time.perf_counter(); x, r = solve_l1(comp); dt = time.perf_counter() - t
        fl = {i for i in range(m) if abs(r[i]) > THRESH}
        p_, r_, f1, fp = prf(fl, corrupt)
        entry["methods"]["orb_l1"] = dict(max_node_error=node_error(comp, graph, x, truth),
                                          precision=p_, recall=r_, f1=f1, false_flags=fp,
                                          n_flagged=len(fl), seconds=dt, finished=True)
        # brute force
        t = time.perf_counter(); x, r, S, evals, ok = solve_brute(comp); dt = time.perf_counter() - t
        if ok:
            p_, r_, f1, fp = prf(S, corrupt)
            entry["methods"]["brute_force"] = dict(max_node_error=node_error(comp, graph, x, truth),
                                                   precision=p_, recall=r_, f1=f1, false_flags=fp,
                                                   n_flagged=len(S), seconds=dt, finished=True, evals=evals)
        else:
            entry["methods"]["brute_force"] = dict(max_node_error=float("nan"), precision=0.0, recall=0.0, f1=0.0,
                                                   false_flags=0, n_flagged=0, seconds=dt, finished=False, evals=evals)
        rows.append(entry)
        print(f"{label:<30} m={m:<3} k={entry['k']}  "
              + "  ".join(f"{k[:5]}: err={v['max_node_error']:.1f} f1={v['f1']:.2f} t={v['seconds']*1000:.0f}ms"
                          + ("" if v["finished"] else " (budget)") for k, v in entry["methods"].items()))
    return rows


def summarize(rows):
    methods = ["least_squares", "orb_l1", "brute_force"]
    out = {}
    for mth in methods:
        vals = [r["methods"][mth] for r in rows]
        done = [v for v in vals if v["finished"]]
        out[mth] = {
            "mean_max_node_error": float(np.mean([v["max_node_error"] for v in done])) if done else float("nan"),
            "median_max_node_error": float(np.median([v["max_node_error"] for v in done])) if done else float("nan"),
            "exact_recovery_rate": float(np.mean([v["max_node_error"] < 1.0 for v in vals])),
            "mean_f1": float(np.mean([v["f1"] for v in vals])),
            "mean_false_flags": float(np.mean([v["false_flags"] for v in vals])),
            "mean_seconds": float(np.mean([v["seconds"] for v in vals])),
            "finished": int(sum(v["finished"] for v in vals)),
            "n": len(vals),
        }
    return out


if __name__ == "__main__":
    rows = run_all()
    summary = summarize(rows)
    (ROOT / "docs" / "benchmark.json").write_text(json.dumps({"scenarios": rows, "summary": summary}, indent=1))
    print("\nSUMMARY over", len(rows), "scenarios")
    print(f"{'method':<15} {'exact-recovery':>14} {'mean max err':>13} {'mean F1':>8} {'false flags':>12} {'mean time':>10}")
    for k, v in summary.items():
        print(f"{k:<15} {v['exact_recovery_rate']*100:>13.0f}% {v['mean_max_node_error']:>13.1f} {v['mean_f1']:>8.2f} "
              f"{v['mean_false_flags']:>12.1f} {v['mean_seconds']*1000:>8.0f}ms")
    if "--chart" in sys.argv:
        from benchmark_chart import render
        render(rows, summary, ROOT / "docs" / "benchmark.png")
        print("wrote docs/benchmark.png")
