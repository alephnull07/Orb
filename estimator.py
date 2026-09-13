"""
estimator.py
------------
Takes claims.json + world.json, builds H matrix, runs three estimators.

The linear model:  y = H @ x + a + epsilon
  x  = final stock at each site  (n_sites unknowns)
  y  = what claims report        (one value per claim)
  H  = maps x to what each claim should read
  a  = corruption (unknown, sparse)
  e  = small honest noise

=== How H is built ===

Stock claim for site s:
  H[i, s] = 1, all others 0
  y[i] = claimed_amount
  Meaning: "x[s] equals this number"

Transfer claim (from F, to T, amount A):
  On a tree, the ONLY way stock enters the subtree rooted at T is
  through the single edge (F→T). So:

    net_flow(F→T) = sum(final_stock[v] for v in subtree(T))
                  - sum(start_stock[v] for v in subtree(T))

  A transfer claim says net_flow(F→T) = A, so:

    sum(x[v] for v in subtree(T)) = A + sum(start[v] for v in subtree(T))

  H row: +1 for every v in subtree(T), 0 elsewhere
  y[i] = A + sum(start[v] for v in subtree(T))

  This is correct and extends to any number of nested transfers.
  The (+1,-1) trick I initially tried only works for leaf targets and
  is wrong for internal nodes that have both inflows and outflows.

Redundancy:
  10 stock claims + 18 transfer claims (9 edges × 2 reporters) = 28 claims
  10 unknowns → 18 degrees of redundancy.
  10 stock claims alone make H full rank — transfers are all overdetermined.
"""

import json
import itertools
from collections import deque
import numpy as np
from scipy.optimize import linprog


def load_world(path="world.json"):
    with open(path) as f:
        return json.load(f)


def load_claims(path="claims.json"):
    with open(path) as f:
        return json.load(f)


def build_tree(edges, root):
    """
    Orient the undirected edge list as a rooted tree.
    Returns children dict: {parent: [child, ...]}
    """
    children = {}
    visited = {root}
    queue = deque([root])
    edge_set = [(f, t) for f, t in edges]

    while queue:
        node = queue.popleft()
        for f, t in edge_set:
            for a, b in [(f, t), (t, f)]:
                if a == node and b not in visited:
                    children.setdefault(a, []).append(b)
                    visited.add(b)
                    queue.append(b)
    return children


def get_subtree(node, children):
    """Return all nodes in the subtree rooted at node (including node itself)."""
    result = [node]
    for child in children.get(node, []):
        result.extend(get_subtree(child, children))
    return result


def build_H(claims, world):
    """
    Build observation matrix H and observation vector y.

    H shape: (n_claims, n_sites)
    y shape: (n_claims,)
    """
    sites = world["sites"]
    start_stock = world["start_stock"]
    edges = world["edges"]
    root = sites[0]  # DEPOT

    n = len(sites)
    site_idx = {s: i for i, s in enumerate(sites)}

    children = build_tree(edges, root)
    # Precompute subtrees for every node
    subtrees = {s: get_subtree(s, children) for s in sites}

    H = np.zeros((len(claims), n))
    y = np.zeros(len(claims))

    for i, c in enumerate(claims):
        if c["type"] == "stock":
            H[i, site_idx[c["site"]]] = 1.0
            y[i] = c["amount"]

        elif c["type"] == "transfer":
            # Transfer (F→T, amount A):
            # sum(x[v] for v in subtree(T)) = A + sum(start[v] for v in subtree(T))
            sub = subtrees[c["to"]]
            for v in sub:
                H[i, site_idx[v]] = 1.0
            start_sum = sum(start_stock[v] for v in sub)
            y[i] = c["amount"] + start_sum

    return H, y


def least_squares(H, y):
    """
    Standard least squares: minimize sum of squared residuals.
    Fails against corruption — smears the lie across all estimates.
    """
    x, _, _, _ = np.linalg.lstsq(H, y, rcond=None)
    residuals = y - H @ x
    return x, residuals


def l1_estimate(H, y, weights=None):
    """
    L1 estimator: minimize sum of |residuals|.

    Reformulation as LP:
      Variables: [x (n_sites), r (n_claims)]
      Minimize:  weights @ r
      Subject to:
        r_i >= y_i - H_i @ x   =>  -H_i @ x - r_i <= -y_i
        r_i >= H_i @ x - y_i   =>   H_i @ x - r_i <= y_i
        r_i >= 0

    Low-confidence claims get lower weight (cheaper to violate).
    """
    m, n = H.shape

    if weights is None:
        weights = np.ones(m)

    c_obj = np.concatenate([np.zeros(n), weights])

    A_ub = np.block([
        [-H, -np.eye(m)],
        [ H, -np.eye(m)]
    ])
    b_ub = np.concatenate([-y, y])

    bounds = [(None, None)] * n + [(0, None)] * m

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if result.status != 0:
        return None, None

    x = result.x[:n]
    residuals = y - H @ x
    return x, residuals


def subset_search(H, y, k, residual_threshold=5.0):
    """
    Exhaustive subset search: ground truth for small k.

    For every combination of k claims to drop:
      - Solve least squares on remaining claims
      - Check if ALL remaining claims agree (|residual| < threshold)
      - Collect consistent stories

    Returns (status, solutions, dropped_sets):
      status = "unique" | "ambiguous" | "none"
    """
    m = H.shape[0]
    n = H.shape[1]
    consistent_solutions = []
    consistent_dropped = []

    for dropped in itertools.combinations(range(m), k):
        kept = [i for i in range(m) if i not in dropped]
        H_sub = H[kept]
        y_sub = y[kept]

        if len(kept) < n:
            continue
        if np.linalg.matrix_rank(H_sub) < n:
            continue

        x_cand, _, _, _ = np.linalg.lstsq(H_sub, y_sub, rcond=None)
        residuals_kept = y_sub - H_sub @ x_cand

        if np.max(np.abs(residuals_kept)) < residual_threshold:
            is_dup = any(
                np.max(np.abs(prev - x_cand)) < 1.0
                for prev in consistent_solutions
            )
            if not is_dup:
                consistent_solutions.append(x_cand)
                consistent_dropped.append(dropped)

    if len(consistent_solutions) == 0:
        return "none", [], []
    elif len(consistent_solutions) == 1:
        return "unique", consistent_solutions, consistent_dropped
    else:
        return "ambiguous", consistent_solutions, consistent_dropped


def flag_claims(claims, residuals, threshold=10.0):
    """Flag claims whose |residual| exceeds threshold."""
    return [
        {"index": i, "source": c.get("source") or c.get("source_str", "?"), "residual": round(float(r), 2)}
        for i, (c, r) in enumerate(zip(claims, residuals))
        if abs(r) > threshold
    ]


def print_results(label, sites, x_est, residuals, claims, truth=None):
    site_idx = {s: i for i, s in enumerate(sites)}
    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"{'='*52}")
    print(f"  {'Site':<10} {'Estimated':>10} {'True':>10} {'Error':>10}")
    for s in sites:
        est = x_est[site_idx[s]]
        if truth:
            true_val = truth[s]
            err = est - true_val
            print(f"  {s:<10} {est:>10.1f} {true_val:>10.1f} {err:>10.1f}")
        else:
            print(f"  {s:<10} {est:>10.1f}")

    flagged = flag_claims(claims, residuals)
    if flagged:
        print(f"\n  Flagged claims (|residual| > 10):")
        for f in flagged:
            print(f"    [{f['index']:02d}] {f['source']}  r={f['residual']}")
    else:
        print(f"\n  No claims flagged.")

    if truth:
        errors = [abs(x_est[site_idx[s]] - truth[s]) for s in sites]
        print(f"\n  Max stock error: {max(errors):.1f}")
        print(f"  Mean stock error: {sum(errors)/len(errors):.1f}")


if __name__ == "__main__":
    world = load_world()
    claims = load_claims()

    sites = world["sites"]
    final_stock = world["final_stock"]

    H, y = build_H(claims, world)
    print(f"H shape: {H.shape}  ({H.shape[0]} claims, {H.shape[1]} sites)")
    print(f"Rank of H: {np.linalg.matrix_rank(H)}")

    x_ls, res_ls = least_squares(H, y)
    print_results("LEAST SQUARES", sites, x_ls, res_ls, claims, truth=final_stock)

    x_l1, res_l1 = l1_estimate(H, y)
    if x_l1 is not None:
        print_results("L1 ESTIMATOR", sites, x_l1, res_l1, claims, truth=final_stock)
    else:
        print("\nL1: solver failed")
