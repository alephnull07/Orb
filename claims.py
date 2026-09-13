"""
claims.py
---------
Generates claims.json from world.json.

Each claim is one "sensor reading" — what a person reported.
Two claim types:
  stock    — a site commander reports their current stock level
  transfer — a driver/receiver reports how much moved between two sites

For each transfer we generate TWO claims: one from the driver (sender side),
one from the receiver. This creates redundancy — if they disagree, L1 can
catch it.

Noise: ±2 on every amount (honest rounding/counting error).
Corruption: k claims are replaced with wildly wrong numbers.
The estimator NEVER sees which claims were corrupted.
"""

import json
import random
import sys


def load_world(path="world.json"):
    with open(path) as f:
        return json.load(f)


def generate_claims(world, seed=0, k=0):
    """
    Returns:
        claims          — list of claim dicts (what the estimator sees)
        corruption_log  — list of corrupted claim indices (estimator never sees this)
    """
    rng = random.Random(seed)
    sites = world["sites"]
    final_stock = world["final_stock"]
    transfers = world["transfers"]

    claims = []

    # --- Stock claims: one per site ---
    # Each site commander reports their current stock.
    for site in sites:
        true_val = final_stock[site]
        noise = rng.randint(-2, 2)
        claims.append({
            "type": "stock",
            "site": site,
            "amount": true_val + noise,
            "confidence": 1.0,
            "source": f"stock_report_{site}"
        })

    # --- Transfer claims: driver + receiver per transfer ---
    # We collapse multiple transfers on the same edge into one net flow per edge
    # so each edge gets exactly two claims (driver + receiver), not two per leg.
    # Net flow per edge: sum up all transfers along that directed edge.
    edge_flows = {}
    for t in transfers:
        key = (t["from"], t["to"])
        edge_flows[key] = edge_flows.get(key, 0) + t["amount"]

    for (frm, to), net_amount in edge_flows.items():
        # Driver claim (sender side)
        noise_d = rng.randint(-2, 2)
        claims.append({
            "type": "transfer",
            "from": frm,
            "to": to,
            "amount": net_amount + noise_d,
            "confidence": 1.0,
            "source": f"driver_{frm}_{to}"
        })
        # Receiver claim (receiver side)
        noise_r = rng.randint(-2, 2)
        claims.append({
            "type": "transfer",
            "from": frm,
            "to": to,
            "amount": net_amount + noise_r,
            "confidence": 1.0,
            "source": f"receiver_{frm}_{to}"
        })

    # --- Plant corruption: replace k claims with wildly wrong numbers ---
    # Pick k distinct indices at random. Use a large wrong value so it's
    # clearly not noise. The estimator never sees corruption_log.
    n = len(claims)
    corrupt_indices = rng.sample(range(n), min(k, n))
    corruption_log = []

    for idx in corrupt_indices:
        original = claims[idx]["amount"]
        # Wrong value: random in [200, 500], far from any realistic reading
        wrong_val = rng.randint(200, 500)
        claims[idx]["amount"] = wrong_val
        corruption_log.append({
            "claim_index": idx,
            "source": claims[idx]["source"],
            "original_amount": original,
            "corrupted_amount": wrong_val
        })

    return claims, corruption_log


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    world = load_world()
    claims, corruption_log = generate_claims(world, seed=seed, k=k)

    with open("claims.json", "w") as f:
        json.dump(claims, f, indent=2)

    # Save corruption log separately — estimator must never load this
    with open("corruption_log.json", "w") as f:
        json.dump(corruption_log, f, indent=2)

    print(f"Generated {len(claims)} claims, {k} corrupted.")
    print(f"Claims -> claims.json")
    print(f"Corruption log (ground truth) -> corruption_log.json\n")

    print("=== CLAIMS ===")
    for i, c in enumerate(claims):
        flag = " <-- CORRUPTED" if any(e["claim_index"] == i for e in corruption_log) else ""
        if c["type"] == "stock":
            print(f"  [{i:02d}] stock    {c['site']:8s}  amount={c['amount']}{flag}")
        else:
            print(f"  [{i:02d}] transfer {c['from']:8s} -> {c['to']:8s}  amount={c['amount']}{flag}")


if __name__ == "__main__":
    main()
