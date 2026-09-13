"""
test_orb.py
-----------
Checkpoint 1 test harness.

Sweeps k = 0, 1, 2, 3 corrupted claims.
Runs least squares and L1 on each.
Reports: k | estimator | max error | mean error | flagged the right claims? | false flags

The checkpoint pass condition:
  - Least squares error grows with k, stays confident (no "I don't know")
  - L1 recovers near-truth at k=1 and k=2, flags the right claims
  - At k=3, L1 starts to struggle — that's the recoverability bound story
"""

import json
import numpy as np
from claims import load_world, generate_claims
from estimator import build_H, least_squares, l1_estimate, flag_claims


def evaluate(x_est, sites, truth):
    site_idx = {s: i for i, s in enumerate(sites)}
    errors = [abs(x_est[site_idx[s]] - truth[s]) for s in sites]
    return max(errors), sum(errors) / len(errors)


def caught_corrupted(flagged_indices, corruption_log, claims):
    """
    Did the estimator flag at least one actually-corrupted claim?
    And how many of its flags were innocent (false positives)?
    """
    corrupt_set = {e["claim_index"] for e in corruption_log}
    flagged_set = set(flagged_indices)

    true_positives = flagged_set & corrupt_set
    false_positives = flagged_set - corrupt_set

    caught = len(true_positives) > 0
    return caught, len(false_positives), list(true_positives), list(false_positives)


def run_trial(k, seed=42):
    world = load_world()
    sites = world["sites"]
    truth = world["final_stock"]

    claims, corruption_log = generate_claims(world, seed=seed, k=k)

    H, y = build_H(claims, world)

    results = []

    # --- Least squares ---
    x_ls, res_ls = least_squares(H, y)
    flagged_ls = flag_claims(claims, res_ls, threshold=10.0)
    flagged_idx_ls = [f["index"] for f in flagged_ls]
    max_err_ls, mean_err_ls = evaluate(x_ls, sites, truth)
    caught_ls, fp_ls, tp_list_ls, fp_list_ls = caught_corrupted(flagged_idx_ls, corruption_log, claims)

    results.append({
        "k": k, "estimator": "least_squares",
        "max_err": max_err_ls, "mean_err": mean_err_ls,
        "caught": caught_ls if k > 0 else "N/A",
        "false_flags": fp_ls,
        "corrupt_indices": [e["claim_index"] for e in corruption_log],
        "flagged_indices": flagged_idx_ls,
    })

    # --- L1 ---
    x_l1, res_l1 = l1_estimate(H, y)
    if x_l1 is not None:
        flagged_l1 = flag_claims(claims, res_l1, threshold=10.0)
        flagged_idx_l1 = [f["index"] for f in flagged_l1]
        max_err_l1, mean_err_l1 = evaluate(x_l1, sites, truth)
        caught_l1, fp_l1, tp_list_l1, fp_list_l1 = caught_corrupted(flagged_idx_l1, corruption_log, claims)

        results.append({
            "k": k, "estimator": "l1",
            "max_err": max_err_l1, "mean_err": mean_err_l1,
            "caught": caught_l1 if k > 0 else "N/A",
            "false_flags": fp_l1,
            "corrupt_indices": [e["claim_index"] for e in corruption_log],
            "flagged_indices": flagged_idx_l1,
        })
    else:
        results.append({
            "k": k, "estimator": "l1", "max_err": None, "mean_err": None,
            "caught": None, "false_flags": None,
        })

    return results, corruption_log


def main():
    print("\n" + "="*80)
    print("  ORB — CHECKPOINT 1 RESULTS")
    print("  Structured claims, seeded corruption, no reader involved")
    print("="*80)

    print(f"\n  {'k':>3}  {'Estimator':>14}  {'MaxErr':>8}  {'MeanErr':>8}  {'Caught?':>8}  {'FalseFlags':>10}")
    print(f"  {'-'*3}  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")

    all_results = []
    for k in [0, 1, 2, 3]:
        results, corruption_log = run_trial(k, seed=42)
        for r in results:
            caught_str = str(r["caught"]) if r["caught"] != "N/A" else "N/A"
            max_err_str = f"{r['max_err']:.1f}" if r["max_err"] is not None else "FAIL"
            mean_err_str = f"{r['mean_err']:.1f}" if r["mean_err"] is not None else "FAIL"
            fp_str = str(r["false_flags"]) if r["false_flags"] is not None else "FAIL"
            print(f"  {r['k']:>3}  {r['estimator']:>14}  {max_err_str:>8}  {mean_err_str:>8}  {caught_str:>8}  {fp_str:>10}")
        all_results.extend(results)
        print()

    print("="*80)
    print("\n  INTERPRETATION")
    print("  k=0: both methods should agree (no corruption, only ±2 noise)")
    print("  k=1: L1 should recover near-truth; LS error should jump")
    print("  k=2: L1 still recovers if redundancy holds; LS gets worse")
    print("  k=3: L1 may start to fail — this is the recoverability boundary")
    print()

    # Print corruption details for each k
    for k in [1, 2, 3]:
        results, corruption_log = run_trial(k, seed=42)
        print(f"  --- k={k} corruption details ---")
        for e in corruption_log:
            print(f"    claim[{e['claim_index']:02d}] {e['source']}: "
                  f"true~{e['original_amount']} → corrupted to {e['corrupted_amount']}")
        l1_res = next(r for r in results if r["estimator"] == "l1")
        print(f"    L1 flagged: {l1_res['flagged_indices']}")
        print(f"    Corrupted:  {l1_res['corrupt_indices']}")
        print()


if __name__ == "__main__":
    main()
