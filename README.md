# ORB

The Datadog for Defense Logistics

**Operational observability for resource networks.**. Typically, defense logistics run on reports from people, sensors, warehouses, vehicles, and disconnected systems. Those reports are often messy, inconsistent, stale, duplicated, or simply wrong.

ORB turns those reports into a single view of the network — showing **what's happening, what the real recovered state is, what's inconsistent, which reports can't be trusted, and when there isn't enough information to know.**

Instead of averaging conflicting data into a confident answer, ORB finds the reports that don't hold together, isolates the source of the inconsistency, and tells you how much corruption the network can tolerate.

**Know what happened. Know what you can trust. Know when you don't know.**


<img width="1720" height="998" alt="image" src="https://github.com/user-attachments/assets/38ef4501-81e5-4f63-b4c8-71df5440c547" />
<img width="1728" height="998" alt="image" src="https://github.com/user-attachments/assets/0afc6140-ecbd-4486-bba0-3742999002df" />


## Headline results

**1,520 runs.** Two domains, corruption from k = 0 to 6, three corruption types, 20 seeds. Corruption support never visible to the estimator; stress cases chosen after the method was fixed.

| | Least squares | ORB (weighted ℓ₁) |
|---|---|---|
| Confidently wrong, any k ≥ 1 | **100%** | **0–10%** (random / directional) |
| Confidently wrong, correlated corruption | 100% | 35–60% at high k |
| Exact recovery at k = 0 | 100% | 100% |
| Exact recovery at k ≥ 1 | **0%** | 70–85% supply, 15–25% water at k = 6 |
| State error vs. truth | 3–10× larger | baseline |
| Detection precision / recall | low / high (flags everything) | **1.00 / 1.00** within the guarantee |

"Confidently wrong" means a single answer was returned with no ambiguity reported, and it was wrong. It is the metric that matters: a wrong answer nobody can distinguish from a right one is worse than no answer.

The correlated-corruption row is the honest one. When a sender and a receiver corroborate the same false transfer, conservation still holds and there is nothing to contradict. That is precisely the case our identifiability analysis predicts is invisible, and the sweep confirms it at the predicted point.

---

## The method

Every report is a sensor reading on a network. The network obeys one constraint: conservation. What leaves one node arrives at another, minus anything consumed or leaked. That turns a pile of inconsistent reports into an over-determined linear system in which corrupted reports produce contradictions that can be located.

```
y = Hx + a + ε
```

- `x` — true state: quantity at each node, flow on each edge, sink per node where losses are allowed
- `y` — the claims extracted from reports
- `H` — which state variables each claim depends on
- `ε` — ordinary noise
- `a` — corruption: sparse, arbitrary magnitude, **support unknown to the estimator**

We solve `min_x Σ wᵢ |yᵢ − Hᵢx|` subject to conservation, as a linear program. Least squares minimizes *squared* error, so it prefers several medium errors to one large one and spreads a lie across every honest report. ℓ₁ is nearly indifferent between those, so it isolates the error on the reports that are actually wrong and fits the rest exactly.

`wᵢ` is reader confidence. For text sources, three extractors read each record independently and the spread across them sets the weight, so low-confidence claims are cheap for the estimator to discard.

---

## Identifiability: what the system knows it cannot know

Before solving, ORB computes the rank of the system and the largest `k` such that deleting any `2k` claim rows still pins down every unknown. That is how many corrupted claims the network can provably survive. When it is zero, ORB says corruption is undetectable on this graph rather than returning a confident answer.

**Three characterized limits, each with the resolving measurement named.**

**1. Coordinated reports are invisible.** A corruption `a = Hc` for some nonzero state shift `c` produces readings identical to an honest network in state `x + c`. Residuals are identically zero. No estimator separates them — not ℓ₁, not least squares, not exhaustive search. Resolution: one independent claim on a node in the support of `c`.

**2. Allowing leaks costs corruption tolerance.** Enabling unknown sinks adds one unknown per node without adding claims, so redundancy drops and `correctable_k` falls to zero. In that regime a node claim that *understates* is mathematically indistinguishable from a real leak: both are explained by inflating the sink. Claims that *overstate* remain detectable, because absorbing them would require a negative sink, which `sink ≥ 0` forbids. A one-directional blind spot with a stated mechanism, not a tuning artifact.

**3. Two sensors, not one.** We expected a single drain meter to resolve limit 2. It does not — that meter and the original claim can both be dropped together, leaving one equation and two unknowns, and `correctable_k` stays at 0. A second independent sensor closes it: any pair dropped still leaves each variable pinned by the other.

| Configuration | correctable_k | Downward corruption |
|---|---|---|
| Sinks disabled | 1 | detected |
| Sinks unknown | 0 | **silently absorbed** |
| Sinks unknown + 1 drain meter | 0 | still absorbed |
| Sinks unknown + 2 independent sensors | 1 | detected |

**Temporal redundancy adds tolerance, not localization.**

| Window | Redundancy | correctable_k |
|---|---|---|
| T = 1 | 0 | 0 |
| T = 2 | 32 | 0 |
| T = 4 | 96 | 1 |

Four snapshots buy one round of corruption tolerance. The state was recoverable at T = 1; what time adds is the ability to *defend* the answer.

---

## Two derived constants, not two tuned ones

**Detection threshold = 5σ of measurement noise.** Honest residuals are bounded by noise propagated through the balance constraints (max observed 4.02 at σ = 1.0); corruption in our generators starts at 50. The gap is structural whenever k ≤ correctable_k, so the threshold sits between them by construction.

| Threshold | Precision | Recall | False positives |
|---|---|---|---|
| 0.5 (original) | 0.222 | 1.000 | 140 |
| 2.0 | 0.548 | 1.000 | 33 |
| 3.0 | 0.870 | 1.000 | 6 |
| **5.0** | **1.000** | **1.000** | **0** |

The original threshold sat *below* the noise floor and flagged 37% of honest claims. A robust alternative (median + n·MAD) peaked at F1 = 0.976 and never matched the fixed rule, because corrupted residuals inflate the median they are being measured against — a real failure of robust statistics when the contamination is what you are estimating from.

**Sink penalty < minimum claim weight.** The ℓ₁ penalty on sink variables holds across four orders of magnitude (1e-4 to 0.5) and breaks at exactly the minimum claim weight (0.8). Below that, absorbing a loss into a sink is cheaper than flagging an honest claim; above it, the reverse. A condition, not a number.

---

## Pipeline

```
files (.csv .tsv .xlsx | .txt .log .md | .jsonl)
   │
   ├─ TABULAR   sample once → one agent call returns a column mapping
   │            → mapping applied row-wise in plain Python (zero LLM calls per row)
   │
   └─ RECORD    split on message boundaries → 3 agents read each record
                independently → spread across readers sets claim weight
   │
   ▼
merge   canonicalize node names across files, dedupe the same event,
        sum distinct events on the same edge
   ▼
compile build H, balance rows, weights; run the identifiability check
   ▼
ℓ₁ estimate → decode: state, flagged claims, provenance to the source line
```

**The agents never build the matrix.** They emit claims in one of three primitives — a node reading, an edge reading, or an aggregate — and the compiler constructs the system. There is no primitive for "leak" or "corrupted." A conclusion can only come from the math, enforced at the contract level and tested.

**A tabular file costs one model call regardless of size.** The agent reads the structure once; applying the mapping is deterministic.

**Adding a domain is configuration, not code.** Node quantities, edge flows, one balance equation per node, and a flag for whether sinks are none, known, or unknown. Supplies and water compile through the identical path.

---

## Verified scenarios

One fuel network (mesh topology — `OP_TALON` is fed by two forward bases), five corruption configurations. Every expected value was derived by solving the system independently before the fixtures were generated.

| Scenario | Planted | Result |
|---|---|---|
| **C** clean | none | all 5 nodes exact, 0 flagged |
| **B** two independent | node −450, edge +400 | all 5 nodes exact, exactly 2 flagged |
| **D** correlated | one reporter −120 across 3 claims | all 5 nodes exact, 3 flagged, every one tracing to that reporter |
| **E** known sinks | node +50, against burn rates of 600–800 | all 5 nodes exact, 1 flagged |
| **F** over-corrupted | 4 claims on a k = 1 graph | **fails as predicted** |

**Scenario F is in the repo on purpose.** The system reported `correctable_k = 1` before solving. Given four corrupted claims it recovered three of five nodes exactly and misattributed the rest: two nodes off by ±350, and an *honest* claim flagged. The failure was predicted quantitatively by the independent solve before the run. Exceeding a stated guarantee has a cost, and this is what that cost looks like.

Scenario D is the informative one: all three flags trace to a single reporter rather than to a location in the network. Errors that cluster by source are the signal a source-level weighting scheme would use.

Scenario B recovers two corruptions on a graph whose guarantee is k = 1. The `2k` condition is a worst-case bound, not a prediction — those two happen to be separable.

---

## Reproduce

```bash
pip install numpy scipy
export ANTHROPIC_API_KEY=...        # or set it in .env

python -m orb.demo tests/fixtures/fuel_B_two_corruptions.csv
python -m orb.metrics                # full sweep -> results/sweep.json
python -m orb.summarize              # markdown table from the sweep
pytest                               # full suite
```

Every run is seeded and deterministic. `results/sweep.json` records the git commit hash, seed list, and timestamp in its meta block.

---

## Evaluation discipline

- **Ground truth is generated, never read.** A test asserts the estimator's input contains no field naming a corrupted claim, a leak location, or a leak magnitude.
- **The method is frozen** at commit `<HASH>`. Cases used for stress testing were chosen after that commit.
- **Simulation with known ground truth is the method, not a fallback.** With real incident data nobody knows the true state, so recovery cannot be scored.
- **We attempted validation on LeakDB** and report what we found: the snapshot we obtained is timestamped before leak onset, so no leak signal is present. We cut the localization claim rather than keep it. What LeakDB did show is that single-snapshot SCADA has zero measurement redundancy, which makes corruption formally undetectable on it — the finding that motivated the temporal analysis above.

---

## What we do worse

- **Correlated corruption degrades us**, to 35–60% confidently wrong at high k on the water domain. Predicted by the identifiability analysis; reported, not hidden.
- **Weighted ℓ₁ can sacrifice cheap honest claims.** In one diagnosed case the estimator flagged a low-weight honest claim instead of the real liar, because abandoning it was cheaper. Classic masking, at a weight boundary we can point to.
- **Past `correctable_k`, misattribution.** Scenario F.
- **The detection floor is not yet measured.** Corruption magnitudes in our generators start well above the noise. How small a lie can be before it hides is the next experiment.
- **Snapshot only.** No temporal dynamics in the estimator; out-of-order arrival is handled by bucketing, not retrodiction.

---

## Prior work

ℓ₁ decoding of sparse errors (Candès & Tao 2005), secure estimation under adversarial attacks and the `2k`-sparse observability condition (Fawzi, Tabuada & Diggavi, IEEE TAC 2014), stealthy false-data injection `a = Hc` (Liu, Ning & Reiter, ACM TISSEC 2011), and conservation-based data reconciliation with gross-error detection (Tamhane & Mah 1985) are established. We build on them directly.

ORB's contribution is applying them over unstructured human reports; modeling the extraction layer as a fallible sensor inside the failure model, with inter-reader disagreement as a confidence weight; characterizing the sink/corruption indistinguishability and its two-sensor resolution on a given graph; and building the abstraction so that a domain is a configuration rather than an implementation.
