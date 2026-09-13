# ORB

**Resilient state estimation for resource distribution networks under corrupted, unstructured reports.**

Aarav & Parth, UCLA CS

---

<img width="1723" height="950" alt="image" src="https://github.com/user-attachments/assets/ac1979f1-754e-40a6-87ed-00f0dfecd12e" />


## The problem

Any system where resources move through a network — defense logistics, disaster relief, utilities, inventory — runs on reports about where things are. Those reports come from people, arrive as unstructured text, and some are wrong: stale, duplicated, or fabricated.

Standard estimation assumes every report is roughly right and averages them. So a single bad report gets quietly absorbed into a confident, wrong answer — and no one can tell the confident answer from a guess. DoD supply-chain visibility sat on GAO's High-Risk List for 29 years for essentially this reason.

ORB recovers the true distribution of resources across the network from the reports that hold together, isolates the ones that don't, and reports when the data genuinely can't distinguish two possibilities instead of guessing.

## The approach

Every report is treated as a sensor reading on a network. The network obeys one constraint: **conservation** — what leaves one node arrives at another; totals don't appear or vanish. This turns a pile of inconsistent reports into an over-determined linear system, where corrupted reports create contradictions that can be located.

We solve it with an **L1 estimator** rather than least squares. Least squares minimizes squared error, so it spreads one large error across every honest report. L1 minimizes absolute error, so it concentrates the error on the few reports that are actually wrong — which matches the real failure model, where a small number of reports are badly corrupted and the rest are fine.

When multiple distributions are equally consistent with the surviving reports, ORB returns the **set** of candidates and flags the ambiguity, rather than committing to one.

## Pipeline

```
unstructured reports  ->  reader (agents)  ->  claims + graph  ->  estimator (L1)  ->  state + flagged reports
```

1. **Reader.** A set of agents parses raw text into structured claims — which node, which quantity, which transfer — and assembles the network graph. Multiple agents run against each other so extraction errors surface as disagreement: the reader itself is a sensor that can be wrong.
2. **Estimator.** Claims become rows of a linear system `y = Hx`. We solve for the state `x` (resource level per node) with a weighted L1 objective, using report confidence as weights. Least squares runs alongside as a baseline.
3. **Output.** The recovered distribution, the reports flagged as inconsistent, and — where the data is insufficient — the set of consistent states.

## The model

```
y = Hx + a + e
```

- `x` — true resource level at each node (hidden)
- `y` — the claims extracted from reports
- `H` — which nodes each claim depends on (stock claim: one node; transfer claim: two)
- `e` — ordinary noise
- `a` — corruption: sparse, arbitrary magnitude, support unknown to the estimator

Estimator: `min_x  Σ_i w_i |y_i − H_i x|`, solved as a linear program. The support of `a` is never given to the estimator.

## What's here

| File | Purpose |
|------|---------|
| `world.py` | Generates a synthetic network with known ground-truth state and transfers. Conservation self-check. |
| `claims.py` | Turns ground truth into reports, with configurable corruption (count, type). Records the corruption for scoring; the estimator never sees it. |
| `reader/` | Agent pipeline: raw text -> structured claims + graph. |
| `estimator.py` | Builds `H`, runs least squares and L1, flags inconsistent reports. |
| `test_orb.py` | End-to-end run and comparison table across corruption levels. |

## Run

```bash
pip install numpy scipy
python test_orb.py
```

Output is a table over `k` (number of corrupted reports): max state error and flagged-report accuracy for least squares vs. L1.

## Evaluation

We generate ground truth, generate reports from it, corrupt a known subset, and measure whether the true state is recovered and whether the corrupted reports are correctly flagged — using synthetic data precisely because it gives known ground truth to score against. Corruption cases used for stress testing are chosen after the method is fixed.

Metrics: state error vs. truth, corrupted-report detection rate, false flags on honest reports, and **false certainty** — how often a single confident answer is returned that is wrong. That last one is the metric we care about most; the point of ORB is that it drops toward zero because ORB returns a set instead of guessing when the data is insufficient.

## Domain-agnostic by construction

Nothing in the estimator depends on what is flowing. Any conserved-flow network reduces to the same `H` and the same objective; only the reader's vocabulary changes. The same engine runs on relief, defense logistics, or any distribution network.

## What's next

- **Coordinated corruption.** Two reports that corroborate each other (a sender and receiver reporting the same false transfer) satisfy conservation and are invisible to the residual test. Characterizing and detecting this is the hard case.
- **Recoverability bound.** Sweep the number of corrupted reports to find where recovery breaks for a given graph, and relate it to redundancy.
- **Trusted-count placement.** Where to spend limited high-confidence verification to protect the most nodes.
- **Advisor.** Given an ambiguity set, name the single measurement that most reduces it and the action that is safe across all candidates.

## Prior work

L1 / robust bad-data detection in state estimation, secure estimation under sparse attacks, and conservation-based data reconciliation are established. ORB's contribution is applying them over unstructured human reports, modeling the LLM reader as a fallible sensor within the failure model, and building the abstraction domain-agnostically.
