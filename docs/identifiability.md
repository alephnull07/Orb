# Identifiability Theory in ORB

This document formalizes the identifiability results implemented in the ORB codebase, with concrete numeric evidence from the test fixtures and test suite (`tests/test_compile.py`, `test_orb.py`, and the temporal stacking module `orb/temporal.py`).

---

## 1. The Observation Model: $a = Hc$ Undetectability

### 1.1 The Linear Model

ORB models every resource-distribution network as a linear observation system:

$$y = Hx + a + \varepsilon$$

where:

| Symbol | Meaning | Shape |
|--------|---------|-------|
| $x$ | Hidden state vector (node quantities, edge flows, unknown sinks) | $(n,)$ |
| $y$ | Observed claim values (what sensors/reporters say) | $(m,)$ |
| $H$ | Observation matrix mapping state to expected observations | $(m \times n)$ |
| $a$ | Corruption vector (sparse, unknown support, unbounded magnitude) | $(m,)$ |
| $\varepsilon$ | Small honest noise ($\pm 2$ in supply domain) | $(m,)$ |

The state vector $x$ is partitioned into three blocks (columns of $H$):

1. **Node quantities** `qty_<node_id>` -- one per node (final stock, demand, pressure).
2. **Edge flows** `flow_<edge_id>` -- one per edge (transfer amount, pipe flow).
3. **Unknown sinks** `sink_<node_id>` -- only for nodes with `sinks="unknown"` (leaks, unmetered losses).

### 1.2 How H is Constructed

Each claim becomes one row of $H$:

- **Node claim** on node $s$: $H_{i,\text{qty}_s} = 1$, all other entries 0. The claim says "the quantity at $s$ is $y_i$."
- **Edge claim** on edge $e$: $H_{i,\text{flow}_e} = 1$, all other entries 0.
- **Sink claim** on node $s$: $H_{i,\text{sink}_s} = 1$ (direct drain meter).
- **Aggregate claim** on nodes $\{s_1, \ldots, s_k\}$: $H_{i,\text{qty}_{s_j}} = 1$ for all $j$. Reports the sum of quantities.

In the earlier supply-chain model (`estimator.py`), transfer claims use subtree sums on a rooted tree:

$$H_{i,v} = 1 \quad \forall v \in \text{subtree}(T), \qquad y_i = A + \sum_{v \in \text{subtree}(T)} \text{start}[v]$$

### 1.3 Hard Balance Constraints

In addition to $H$, the compiler produces equality constraints $A_{\text{eq}} x = b_{\text{eq}}$ encoding conservation at every node:

$$\text{qty}_i - \sum_{j \in \text{in}(i)} \text{flow}_j + \sum_{j \in \text{out}(i)} \text{flow}_j + \text{sink}_i = \text{initial}_i$$

The combined system $[H; A_{\text{eq}}]$ must have full column rank for identifiability.

### 1.4 The Undetectability Condition

A corruption vector $a$ is **undetectable** if and only if $a = Hc$ for some nonzero state shift $c$. In that case, the corrupted observations $y' = H(x + c) + \varepsilon$ look identical to a legitimate state $x' = x + c$ with zero corruption. The residual $y' - Hx' = 0$, and no estimator -- regardless of computational power -- can distinguish the two.

**Concrete instance** (from `ORB_RESEARCH_CONTEXT.md`): In the supply chain, if both a sender and receiver report the same false transfer amount, this satisfies conservation and is invisible. The corruption $a = Hc$ where $c$ shifts the sender's stock down and the receiver's stock up by the same amount. Resolution requires a third independent claim on one of the involved nodes.

---

## 2. The 2k-Row-Deletion Recoverability Condition

### 2.1 Formal Statement

The function `_correctable_k()` in `orb/compile.py` (lines 176--201) computes the largest integer $k$ such that:

> **For every subset of $2k$ rows deleted from $H$, the matrix $[H_{\text{remaining}}; A_{\text{eq}}]$ retains full column rank $= n_{\text{vars}}$.**

This is the discrete, exhaustive version of the Candes-Tao / Fawzi-Tabuada-Diggavi recoverability condition: $k$ corrupted claims can be exactly identified if and only if the 2k-row-deletion property holds. The factor of 2 accounts for the worst case: $k$ rows are corrupted (unknowns), and identifying which $k$ are corrupt requires distinguishing them from any other $k$ rows, giving $2k$ total.

### 2.2 Test Results

#### toy_supply.json (no sinks)

| Property | Value |
|----------|-------|
| $H$ shape | $7 \times 7$ |
| $A_{\text{eq}}$ shape | $4 \times 7$ |
| $n_{\text{vars}}$ | 7 |
| $m$ (claims) | 7 |
| rank $[H; A_{\text{eq}}]$ | 7 (full rank) |
| correctable_k | **1** |

With 4 balance constraints from $A_{\text{eq}}$ contributing rank, only 3 additional rows from $H$ are needed to reach full rank 7. Dropping any 2 rows of $H$ (for $k = 1$) always leaves $\geq 5$ rows in $H$, which combined with $A_{\text{eq}}$'s 4 rows maintains full rank. But at $k = 2$ (dropping 4 rows from 7), certain combinations fail.

#### toy_water_sinks.json (with unknown sink, no extra sensors)

| Property | Value |
|----------|-------|
| $H$ shape | $7 \times 8$ |
| $A_{\text{eq}}$ shape | $4 \times 8$ |
| $n_{\text{vars}}$ | 8 (7 + 1 sink variable) |
| $m$ (claims) | 7 |
| rank $[H; A_{\text{eq}}]$ | 8 (full rank) |
| correctable_k | **0** |
| redundancy ($m - n$) | $-1$ |

Adding the unknown sink `sink_BASE_B` introduces an 8th variable. With only 7 claims, the system has **negative redundancy** ($m - n = -1$). It is identifiable only because the 4 balance constraints supply the missing rank. But there is no margin: dropping any 2 rows of $H$ causes rank deficiency. The system cannot tolerate even a single corruption.

#### toy_water_metered.json (with two extra sensors)

| Property | Value |
|----------|-------|
| $H$ shape | $9 \times 8$ |
| $A_{\text{eq}}$ shape | $4 \times 8$ |
| $n_{\text{vars}}$ | 8 |
| $m$ (claims) | 9 |
| rank $[H; A_{\text{eq}}]$ | 8 (full rank) |
| correctable_k | **1** |
| redundancy ($m - n$) | $+1$ |

Two additional sensors (m1: drain meter on `sink_BASE_B`, m2: level sensor on `qty_BASE_B`) push redundancy to $+1$, and correctable_k rises from 0 to 1.

#### Supply chain world (estimator.py, no A_eq)

| Property | Value |
|----------|-------|
| $H$ shape | $28 \times 10$ |
| $n$ (sites) | 10 |
| rank $H$ | 10 (full rank) |
| redundancy ($m - n$) | 18 |

10 stock claims + 18 transfer claims (9 edges x 2 reporters) yield 18 degrees of redundancy. When checking $H$-only row deletion (no $A_{\text{eq}}$), the formal $k = 0$ because dropping the DEPOT stock claim (row 0) plus any other row makes $H$ rank-deficient -- the DEPOT column has only one direct observation. All 27 failing pairs involve the DEPOT stock claim.

Despite the formal $k = 0$ on $H$-alone, L1 recovery in practice succeeds at $k = 1, 2,$ and even $3$:

| $k$ | LS max error | LS mean error | L1 max error | L1 mean error | L1 caught? | L1 false positives |
|-----|-------------|--------------|-------------|--------------|-----------|-------------------|
| 0 | 2.0 | 0.9 | 2.0 | 1.2 | N/A | 0 |
| 1 | 69.4 | 12.8 | 2.0 | 1.2 | Yes | 0 |
| 2 | 176.3 | 31.3 | 2.0 | 1.6 | Yes | 0 |
| 3 | 202.5 | 49.3 | 2.0 | 1.6 | Yes | 0 |

At $k = 1$: L1 identifies the exact corrupted claim (index 24) with zero false positives, while least squares smears the error across all estimates (max error 69.4 vs 2.0). At $k = 3$: L1 still recovers and correctly identifies all three corrupted claims $\{5, 24, 25\}$, but this is at the theoretical boundary -- the 2k-row condition is not universally satisfied, so success depends on which specific claims are corrupted.

---

## 3. Sink/Corruption Confusability

### 3.1 The Structural Problem

When a node has `sinks="unknown"`, the balance constraint at that node includes the sink variable:

$$\text{qty}_i + \text{sink}_i + \sum_{\text{out}} \text{flow}_j - \sum_{\text{in}} \text{flow}_j = \text{initial}_i$$

A **downward** corruption on the node's quantity claim (reporting less than the true value) is structurally indistinguishable from an increased leak. If the true state has $\text{qty}_{\text{B}} = 380$ and $\text{sink}_{\text{B}} = 20$, but the claim reports $\text{qty}_{\text{B}} = 330$, the LP can explain this by setting $\text{qty}_{\text{B}} = 330$ and $\text{sink}_{\text{B}} = 70$ -- a valid solution that satisfies all constraints with zero residuals.

### 3.2 Direction Asymmetry

This confusability is **one-directional** (`test_direction_asymmetry_c6` in `tests/test_compile.py`):

**Downward corruption** (claim reports *less* than truth):

- The LP absorbs the discrepancy by inflating `sink_BASE_B`.
- Zero claims flagged. The estimator is **falsely certain**.
- Tested: $c_6$ corrupted from 380 to 330 yields `sink_BASE_B = 70.0` (inflated from true 20.0), zero flags.
- Tested: $c_6$ corrupted from 380 to 230 yields `sink_BASE_B > 25`, zero flags.

**Upward corruption** (claim reports *more* than truth):

- To absorb it, the sink would need to go negative, but $\text{sink}_i \geq 0$ is enforced.
- The LP flags $c_6$ as the cheapest inconsistency.
- Tested: $c_6$ corrupted from 380 to 530 yields `sink_BASE_B = 0.0`, and $c_6$ is flagged.

**Edge claims are detectable in both directions**: Corrupting edge claim $c_3$ (flow on $e_2$) by $\pm 150$ is flagged regardless of direction, because edge flow variables do not interact with the sink variable in the balance equation.

### 3.3 The False Certainty Problem

The `correctable_k = 0` on `toy_water_sinks.json` is the formal indicator: the system cannot guarantee correction of even one corruption when sinks are unknown. The confusability is not a bug in the LP -- it is the correct optimum under the L1 objective. The LP prefers the cheaper explanation (a sink costs $\lambda_{\text{sink}} \times \text{amount}$ per unit vs. $w_i \times |\text{residual}|$ for flagging), and since $\lambda_{\text{sink}} = 0.01 \ll w_i = 0.8$, absorbing into the sink is always preferred.

This is precisely the "false certainty" the hackathon brief asks to expose: the estimator returns a plausible state with zero flagged claims, but the state is wrong.

### 3.4 Lambda-Sink Sensitivity

The break-even point between absorbing a discrepancy as a sink vs. flagging it as corruption is at $\lambda_{\text{sink}} = w_{\text{claim}}$. From `test_lambda_sink_sensitivity`:

| $\lambda_{\text{sink}}$ | Behavior | `sink_BASE_B` | Claims flagged |
|--------------------------|----------|---------------|----------------|
| 0.0001 -- 0.5 | Correct: sink absorbs true leak | $\approx 20$ | None |
| 1.0 -- 10.0 | Wrong regime: sink suppressed | $< 1.0$ | Yes (truthful claims flagged) |

The valid operating range is $\lambda_{\text{sink}} < \min(w_i) = 0.8$.

---

## 4. The Two-Sensors-Not-One Result

### 4.1 The Problem

Adding a single independent sensor to a sink-enabled node is insufficient to achieve `correctable_k >= 1`. Two sensors of different types are required.

### 4.2 Why One Sensor Fails

**Adding only m1** (drain meter, observes `sink_BASE_B` directly):

- $H$ becomes $8 \times 8$, $A_{\text{eq}}$ remains $4 \times 8$.
- `correctable_k = 0`.
- Failure mode: dropping the pair $\{c_6, m_1\}$ removes both independent observations of `BASE_B`. The only remaining constraint on $\{\text{qty}_{\text{BASE\_B}}, \text{sink}_{\text{BASE\_B}}\}$ is the single balance row -- one equation in two unknowns. Rank drops from 8 to 7.

**Adding only m2** (level sensor, second observation of `qty_BASE_B`):

- $H$ becomes $8 \times 8$, `correctable_k = 0`.
- Failure mode: dropping the pair $\{c_6, m_2\}$ removes both observations of `qty_BASE_B`. The `sink_BASE_B` column is now determined only through the balance constraint, but with `qty_BASE_B` underdetermined, so is the sink. Rank drops from 8 to 7.
- Additionally, dropping $\{c_1, c_3\}$ (DEPOT node claim + flow on $e_2$) also causes rank deficiency.

### 4.3 Why Two Sensors Suffice

**Adding both m1 and m2** (`toy_water_metered.json`):

- $H$ becomes $9 \times 8$, `correctable_k = 1`.
- No pair of dropped $H$ rows can simultaneously eliminate all paths to both `qty_BASE_B` and `sink_BASE_B`:
  - Drop $\{c_6, m_1\}$: m2 still observes `qty_BASE_B`, and balance pins `sink_BASE_B`. Full rank.
  - Drop $\{c_6, m_2\}$: m1 still observes `sink_BASE_B`, and balance pins `qty_BASE_B`. Full rank.
  - Drop $\{m_1, m_2\}$: $c_6$ observes `qty_BASE_B`, balance pins `sink_BASE_B`. Full rank.

The two sensors provide **independent paths** to the two unknowns ($\text{qty}_{\text{BASE\_B}}$ and $\text{sink}_{\text{BASE\_B}}$) at the sink-enabled node. With one sensor, there always exists a pair of claims whose removal leaves a $2 \times 1$ underdetermined subsystem at that node.

### 4.4 Metered Detection of Previously-Invisible Corruption

With both sensors installed, the previously invisible downward corruption on $c_6$ becomes detectable (`test_metered_detects_downward_corruption`):

| Configuration | $c_6$ corrupted $380 \to 330$ | `sink_BASE_B` | Flagged |
|---|---|---|---|
| Unmetered (7 claims) | Absorbed silently | 70.0 (inflated) | None |
| Metered (9 claims, m1 + m2) | Detected | 20.0 (correct) | $c_6$ flagged |

The drain meter m1 reports `sink = 20`, contradicting any inflated-sink explanation. The level sensor m2 reports `qty = 380`, contradicting $c_6$'s false 330. The LP resolves the inconsistency by flagging $c_6$ as the cheapest claim to discard.

---

## 5. Redundancy vs. Window-Length Tradeoff

### 5.1 Temporal Stacking

The module `orb/temporal.py` exploits the assumption that the leak (sink) is constant across multiple SCADA snapshots. With $T$ timestamps, the state vector is:

$$x = [\underbrace{\text{qty}^{(1)}, \ldots, \text{qty}^{(T)}}_{T \times N} \;|\; \underbrace{\text{flow}^{(1)}, \ldots, \text{flow}^{(T)}}_{T \times E} \;|\; \underbrace{\text{sink}}_{N}]$$

Node quantities and edge flows are per-timestamp ($T \times (N + E)$ variables), but the $N$ sink variables are **shared across all timestamps**. Each snapshot contributes $N + E$ claims and $N$ balance constraints.

### 5.2 Numeric Results: Small Network (3 nodes, 3 edges)

| $T$ | $n_{\text{vars}}$ | $m$ (claims) | balance rows | redundancy | correctable_k |
|-----|-------------------|-------------|-------------|-----------|---------------|
| 1 | 9 | 6 | 3 | 0 | 0 |
| 2 | 15 | 12 | 6 | 3 | 0 |
| 3 | 21 | 18 | 9 | 6 | **1** |
| 4 | 27 | 24 | 12 | 9 | 1 |
| 5 | 33 | 30 | 15 | 12 | 1 |
| 6 | 39 | 36 | 18 | 15 | 1 |

### 5.3 Numeric Results: Water Network (7 nodes, 9 edges)

| $T$ | $n_{\text{vars}}$ | $m$ (claims) | balance rows | redundancy | correctable_k |
|-----|-------------------|-------------|-------------|-----------|---------------|
| 1 | 23 | 16 | 7 | 0 | 0 |
| 2 | 39 | 32 | 14 | 7 | 0 |
| 3 | 55 | 48 | 21 | 14 | **1** |
| 4 | 71 | 64 | 28 | 21 | 1 |
| 5 | 87 | 80 | 35 | 28 | 1 |
| 6 | 103 | 96 | 42 | 35 | 1 |
| 8 | 135 | 128 | 56 | 49 | 1 |
| 10 | 167 | 160 | 70 | 63 | 1 |

### 5.4 Analysis

**Growth rates per additional snapshot:**

- Variables grow by $N + E$ (new per-timestamp quantities and flows, but sinks are shared).
- Claims grow by $N + E$ (one observation per sensor per timestamp).
- Balance rows grow by $N$.
- Effective redundancy grows by: $(N + E) + N - (N + E) = N$ per additional snapshot.

For the 7-node water network, each snapshot adds $N = 7$ units of redundancy. At $T = 1$, redundancy is 0; at $T = 2$, redundancy is 7; at $T = 3$, redundancy reaches 14 and `correctable_k` rises to 1.

**The threshold is $T = 3$** for both tested topologies. Despite further redundancy growth beyond $T = 3$, `correctable_k` remains at 1 through $T = 10$ because the exhaustive combinatorial search (capped at $\max_k = 4$) finds that specific claim-pair deletions still break rank. The shared sink variables create a structural bottleneck: no matter how many timestamps are added, the sink variables are observed only through the balance constraints and any explicit sink meters, so the vulnerability at sink-enabled nodes persists.

**Redundancy alone does not determine correctable_k.** At $T = 2$ with the 7-node network, redundancy is 7, yet `correctable_k = 0`. At $T = 3$, redundancy is 14 and `correctable_k = 1`. The extra redundancy must be **structurally diverse** -- it must cover different columns of the observation matrix -- not merely duplicative.

---

## 6. Demo Water Network Results

The `water_world.py` module generates a 6-junction mesh network (ring topology + chord edge + 2 reservoir feed pipes) with a planted 300-unit leak at junction J_03.

### 6.1 Configuration

| Parameter | Value |
|-----------|-------|
| Junctions | 6 (J_01 through J_06) |
| Reservoir | R_01 (feeds J_01 and J_04) |
| Pipes | 9 (7 internal ring+chord, 2 reservoir) |
| Leak node | J_03 |
| Leak size | 300.0 units |
| Base demand | ~100 per junction ($\pm 20$ jitter) |
| Sensor fraction | 1.0 (all sensors active) |
| Extra demand sensors | 6 (redundant demand readings) |
| Sink meters | All 6 junctions |
| Noise $\sigma$ | 1.0 |

### 6.2 Results

| Property | Value |
|----------|-------|
| $n_{\text{vars}}$ | 22 |
| $m$ (claims) | 29 |
| rank $[H; A_{\text{eq}}]$ | 22 (full rank) |
| correctable_k | **1** |
| True leak rank | **#1** |
| Top sink: J_03 | 300.54 |
| Second sink: J_02 | 0.74 |
| Margin (top - second) | **299.8** |

The L1 estimator correctly localizes the 300-unit leak at J_03 with a margin of 299.8 over the next-highest sink. The `correctable_k >= 1` requirement is satisfied by the combination of redundant demand sensors and sink meters at every junction, ensuring that each junction has at least two independent observations.

---

## 7. Summary of Key Results

| Result | Where tested | Finding |
|--------|-------------|---------|
| Undetectability | `ORB_RESEARCH_CONTEXT.md`, `estimator.py` | $a = Hc$ corruptions are invisible to all estimators |
| 2k-row-deletion | `_correctable_k()`, `test_identifiability` | $k$ corruptions correctable iff every 2k-row deletion of $H$ preserves rank of $[H_{\text{rem}}; A_{\text{eq}}]$ |
| Sink confusability | `test_sink_corruption_confusability` | Downward corruption on sink-enabled nodes is absorbed as inflated leak (false certainty) |
| Direction asymmetry | `test_direction_asymmetry_c6` | Downward lies invisible, upward lies detected (sink $\geq 0$ constraint) |
| Two-sensors-not-one | `test_metered_correctable_k` | One sensor leaves correctable_k = 0; two sensors of different types raise it to 1 |
| Temporal stacking | `orb/temporal.py` | $T = 3$ snapshots needed for correctable_k = 1; redundancy grows by $N$ per snapshot but structural diversity is the binding constraint |
| L1 vs. LS | `test_orb.py` | L1 recovers truth at $k = 1, 2, 3$ corruptions (max error 2.0); LS smears error (max error up to 202.5 at $k = 3$) |

---

## References

- Candes & Tao, "Decoding by Linear Programming," IEEE Trans. Inf. Theory 2005.
- Fawzi, Tabuada, Diggavi, "Secure Estimation and Control for Cyber-Physical Systems Under Adversarial Attacks," IEEE TAC 2014.
- Liu, Ning, Reiter, "False Data Injection Attacks against State Estimation in Electric Power Grids," ACM TISSEC 2011.
- Tamhane & Mah, data reconciliation / gross error detection, Technometrics 1985.
