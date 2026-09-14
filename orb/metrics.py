"""
orb/metrics.py
--------------
Metrics harness: sweep over generators, corruption types, estimators, and seeds.
NO LLM calls — runs on structured claims from world generators directly.

Usage:
    python -m orb.metrics
"""

from __future__ import annotations

import json
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .compile   import compile as compile_graph
from .decode    import flag_threshold
from .run_graph import _l1_solve


# ---------------------------------------------------------------------------
# Least-squares solver (uses the same compiled H, y, A_eq, b_eq)
# ---------------------------------------------------------------------------

def _ls_solve(compiled: dict) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least-squares with hard balance constraints via KKT."""
    H      = compiled["H"]
    y      = compiled["y"]
    w      = compiled["w"]
    A_eq   = compiled["A_eq"]
    b_eq   = compiled["b_eq"]

    m, n = H.shape
    p    = A_eq.shape[0]

    # Weighted H and y:  W^{1/2} H x ≈ W^{1/2} y
    W_sqrt = np.diag(np.sqrt(w))
    Hw = W_sqrt @ H
    yw = W_sqrt @ y

    # KKT system for equality-constrained WLS:
    # [ Hw'Hw   A_eq' ] [x]     = [ Hw'yw ]
    # [ A_eq    0     ] [lam]     [ b_eq  ]
    HtH = Hw.T @ Hw
    Hty = Hw.T @ yw

    KKT = np.zeros((n + p, n + p))
    KKT[:n, :n] = HtH
    KKT[:n, n:] = A_eq.T
    KKT[n:, :n] = A_eq

    rhs = np.zeros(n + p)
    rhs[:n] = Hty
    rhs[n:] = b_eq

    try:
        sol = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError:
        # Fallback to lstsq if singular
        sol, _, _, _ = np.linalg.lstsq(KKT, rhs, rcond=None)

    x_hat = sol[:n]
    residuals = y - H @ x_hat
    return x_hat, residuals


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    x_hat: np.ndarray,
    residuals: np.ndarray,
    compiled: dict,
    true_state: np.ndarray,
    corrupt_mask: list[bool],
    threshold: float | None = None,
) -> dict:
    """
    Compute all metrics for a single run.

    Returns dict with:
      false_certainty, exact_recovery, state_error_max, state_error_mean,
      detection_precision, detection_recall, correctable_k
    """
    n_vars = len(true_state)

    # ── state error ─────────────────────────────────────────────────────────
    # x_hat may have more entries than true_state if compile adds extra vars
    # but true_state was built to match compile's layout
    x_compare = x_hat[:n_vars] if len(x_hat) >= n_vars else x_hat
    t_compare = true_state[:len(x_compare)]

    state_err = np.abs(x_compare - t_compare)
    state_error_max = float(np.max(state_err)) if len(state_err) > 0 else 0.0
    state_error_mean = float(np.mean(state_err)) if len(state_err) > 0 else 0.0

    # ── exact recovery ──────────────────────────────────────────────────────
    exact_recovery = 1 if state_error_max < 5.0 else 0

    # ── detection metrics ───────────────────────────────────────────────────
    if threshold is None:
        _, threshold = flag_threshold(residuals)
    flagged = np.abs(residuals) > threshold
    n_claims = len(corrupt_mask)

    # Ensure same length
    flagged_arr = flagged[:n_claims] if len(flagged) >= n_claims else np.pad(
        flagged, (0, n_claims - len(flagged)), constant_values=False
    )
    mask_arr = np.array(corrupt_mask[:len(flagged_arr)])

    tp = int(np.sum(flagged_arr & mask_arr))
    fp = int(np.sum(flagged_arr & ~mask_arr))
    fn = int(np.sum(~flagged_arr & mask_arr))

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if sum(corrupt_mask) == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # no corruptions → perfect recall

    # ── false certainty ─────────────────────────────────────────────────────
    # FC = 1 when the estimator is wrong AND does not correctly identify
    # the corruption.  "Correctly" means precision ≥ 0.5: more than half
    # the flags point at real problems.  An estimator that flags everything
    # (LS smearing) has precision ≪ 0.5 and gets FC = 1 whenever wrong.
    # L1 concentrates residuals on corrupt claims → precision ≈ 1 when
    # k ≤ correctable_k → FC = 0.
    is_wrong = (state_error_max >= 5.0)
    has_corruption = any(corrupt_mask)
    flags_informative = (tp > 0) and (precision >= 0.5)
    false_certainty = 1 if (is_wrong and (not has_corruption or not flags_informative)) else 0

    # ── correctable_k from compile ──────────────────────────────────────────
    correctable_k = compiled["report"]["correctable_k"]

    return {
        "false_certainty":     false_certainty,
        "exact_recovery":      exact_recovery,
        "state_error_max":     round(state_error_max, 4),
        "state_error_mean":    round(state_error_mean, 4),
        "detection_precision": round(precision, 4),
        "detection_recall":    round(recall, 4),
        "correctable_k":       correctable_k,
    }


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(
    generator: str,
    k: int,
    estimator: str,
    corruption_type: str,
    seed: int,
) -> dict:
    """Run one combination and return metrics + timing."""
    from .supply_world import generate_supply_graph
    from .water_world import generate_water_graph

    t0 = time.perf_counter_ns()

    # ── generate ────────────────────────────────────────────────────────────
    if generator == "supply":
        graph, truth = generate_supply_graph(
            seed=seed, n_sites=6, k=k, corruption_type=corruption_type,
        )
    elif generator == "water":
        graph, truth = generate_water_graph(
            seed=seed, n_junctions=6, leak_idx=2, leak_size=300.0,
            k=k, corruption_type=corruption_type,
        )
    else:
        raise ValueError(f"Unknown generator: {generator}")

    # ── compile ─────────────────────────────────────────────────────────────
    compiled = compile_graph(graph)

    # ── solve ───────────────────────────────────────────────────────────────
    if estimator == "l1":
        x_hat, residuals = _l1_solve(compiled)
    elif estimator == "least_squares":
        x_hat, residuals = _ls_solve(compiled)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    t1 = time.perf_counter_ns()
    compute_ms = round((t1 - t0) / 1e6, 2)

    # ── metrics ─────────────────────────────────────────────────────────────
    metrics = compute_metrics(
        x_hat, residuals, compiled,
        truth["true_state"], truth["corrupt_mask"],
    )
    metrics["compute_ms"] = compute_ms

    return {
        "generator":       generator,
        "k":               k,
        "estimator":       estimator,
        "corruption_type": corruption_type,
        "seed":            seed,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------

def run_sweep(
    generators: list[str] | None = None,
    k_range: list[int] | None = None,
    estimators: list[str] | None = None,
    corruption_types: list[str] | None = None,
    seeds: list[int] | None = None,
    output_path: str = "results/sweep.json",
) -> dict:
    """
    Run the full parameter sweep.

    Returns the sweep dict (also written to output_path).
    """
    if generators is None:
        generators = ["supply", "water"]
    if k_range is None:
        k_range = list(range(7))  # 0..6
    if estimators is None:
        estimators = ["least_squares", "l1"]
    if corruption_types is None:
        corruption_types = ["random", "correlated", "directional"]
    if seeds is None:
        seeds = list(range(20))

    # Git hash
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    meta = {
        "git_hash":   git_hash,
        "seeds":      seeds,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "generators": generators,
        "k_range":    k_range,
        "estimators": estimators,
        "corruption_types": corruption_types,
    }

    rows = []
    total = len(generators) * len(k_range) * len(estimators) * len(corruption_types) * len(seeds)
    done = 0

    for gen in generators:
        for k in k_range:
            for est in estimators:
                for ct in corruption_types:
                    # Skip corruption_type variation when k=0 (no corruption)
                    if k == 0 and ct != corruption_types[0]:
                        done += len(seeds)
                        continue
                    for s in seeds:
                        row = run_single(gen, k, est, ct, s)
                        rows.append(row)
                        done += 1

    sweep = {"meta": meta, "rows": rows}

    # Write output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(sweep, fh, indent=2, default=_json_default)

    print(f"[metrics] {len(rows)} runs written to {out}")
    return sweep


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sweep = run_sweep()
    # Print summary
    from .summarize import print_summary
    print_summary(sweep)
