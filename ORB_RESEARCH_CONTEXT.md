# ORB — Research Context and the Graph Generator Direction

Context handoff for continuing development. Read fully before touching the graph builder.

---

## 1. What ORB is

ORB recovers the true state of a resource-distribution network from unstructured human reports, some of which are wrong, and says when it can't.

Pipeline, as it exists now:

```
unstructured reports (text, radio logs, field messages)
  -> 3 LLM readers, run independently
  -> consensus across readers -> structured claims
  -> graph builder -> nodes, edges, measurement matrix H
  -> L1 estimator (linear program) against a conservation constraint
  -> recovered state per node + flagged inconsistent reports
  -> frontend
```

Validated on: 10 synthetic supply-drop networks with planted corruption (all recovered), and LeakDB (real water distribution leak data), run through the same estimator with no changes other than parser vocabulary. Leaking junctions located in all 10 scenarios.

Hackathon track brief (Defense, "Resilient estimation for critical infrastructure"): `y = Hx + a + ε`, where `x` is hidden state, `H` maps state to what each report should say, `ε` is small noise, `a` is sparse corruption of unknown support. Brief rewards: recoverability bounds, sensor placement, comparing recovery strategies, exposing hidden failure modes, and "proofs of impossibility that identify the missing information." Brief explicitly says a polished interface alone does not establish a claim, and wants false certainty reported as a metric.

---

## 2. The research path, in order

### 2a. Why L1 and not least squares
Least squares minimizes squared residuals, so it prefers several medium errors over one large one. That spreads a single lie across every honest report, and the largest residual frequently lands on an innocent report (masking). Always returns one confident answer.

L1 minimizes absolute residuals. Nearly indifferent between one large residual and several small ones, so it isolates the lie on the report that's actually wrong and fits the rest exactly. Solved as an LP via slack variables: `min Σ w_i t_i  s.t.  -t ≤ y - Hx ≤ t`. `scipy.optimize.linprog(method="highs")`. Under enough redundancy L1 returns the same answer as exhaustive subset search (Candès–Tao 2005; Fawzi–Tabuada–Diggavi, IEEE TAC 2014).

### 2b. Identifiability — when recovery is impossible
A corruption `a` is undetectable iff `a = Hc` for some state shift `c`: it looks exactly like a legitimate state change, residuals are zero, no estimator can see it. More compute doesn't help; only an additional independent reading does. Recoverability condition: `k` lies are correctable iff deleting ANY `2k` rows of `H` leaves it full column rank. Redundancy `m - n` bounds `k`; placement decides which lies are catchable.

Concrete instance in the supply domain: a sender and receiver both reporting the same false transfer satisfies conservation and is invisible. Resolved only by a third independent claim on one of the nodes.

### 2c. What's established vs. what's ours (be honest about this)
Established: L1 bad-data detection (power systems, 1970s); secure estimation under sparse attacks (FTD 2014); conservation-based data reconciliation with gross-error detection (chemical engineering, 1980s; Tamhane & Mah 1985); greedy sensor placement; LLM structured extraction.

Ours (literature check found no prior unifying work):
1. **The LLM reader modeled as a corruptible sensor** inside the `y = Hx + a + ε` model. Extraction error is corruption injected upstream of the estimator.
2. **Reader disagreement as a residual.** Multiple readers; where they disagree becomes a confidence weight `w_i = 1/(spread_i + λ)` on that claim's row in weighted L1. Same math one level up. (NOT YET DONE — currently we take consensus, which throws this signal away. This is the next estimator change.)
3. **Domain-agnostic abstraction**: any conserved-flow network reduces to the same `H` and the same objective. Proven by the LeakDB run.
4. Graph-specific identifiability characterization: enumerate the exact coordinated report supports satisfying `a = Hc` on a given graph, and name the single trusted reading that resolves each.

### 2d. Competing project on the same track (for framing only, do not name them)
A fire-spread estimator: heuristic consistency rules with ~15 tuned thresholds, candidate-set enumeration, tolerates exactly one liar (`k=1` hardcoded), only tolerates lies in one direction (readings colder than predicted), no `H`, no identifiability result, but strong engineering discipline and a real metrics harness. Our differentiators: convex estimator with a recovery guarantee, symmetric lie handling, multiple simultaneous corruptions, identifiability map, unstructured-text ingestion, generality. Our gap: evidence and metrics discipline. Close it.

---

## 3. The graph generator direction

### 3a. The claim we want to be able to make
A fire in a building, supplies across bases, water in pipes, cash across accounts: these are not different systems. They are different **plan files**. The reader, the L1 estimator, and the ambiguity reporting are shared. What changes per domain is declared, not coded.

### 3b. What "declare the constraint, not the domain" means concretely
A plan/schema specifies:

- **Node types** and what hidden quantity each holds (stock, pressure, temperature, balance).
- **Edge types** and what they carry (transfer, pipe flow, heat path, transaction).
- **The physical rule** that constrains readings. This is the thing that generates rows of `H`:
  - Conservation of stock: transfer out of A = transfer into B; totals don't appear or vanish.
  - Pressure/flow for water: flow along a pipe is a function of pressure difference; junction inflow = outflow (plus leak term).
  - Heat transfer for fire: temperature change in a room depends on neighbors' temperatures and edge rates, **plus a source term** when the room is burning. Not strictly conserved — the builder must support sources.
- **Reader vocabulary**: the domain nouns and claim types the extractors should produce.
- **Trusted references**: which readings are high-confidence (a depot manifest, a careful count site, a calibrated sensor).

### 3c. What the current builder assumes that has to be relaxed
- Tree-shaped supply network (depot → base → outpost). Generalize to arbitrary graphs.
- Strict conservation with no sources or sinks. Add a source/sink term per node so fire (heat generation) and water (leaks as unknown sinks) fit. LeakDB already forced a partial version of this; make it explicit in the schema.
- Snapshot only. Time-varying dynamics (`x_{t+1} = A x_t + ...`) are a later extension. The brief says to add time only after stating what temporal information adds. Keep snapshot as the default; design the schema so a time step can be added without rewriting.
- Reader vocabulary hardcoded for supplies. Move to the plan.

### 3d. Design target for the builder
```
plan = {
  nodes: [{id, type, state_var}],
  edges: [{from, to, type}],
  constraint: {
    kind: "conservation" | "flow_pressure" | "diffusion_with_source",
    params: {...}
  },
  claim_types: [...],          # what the reader may emit
  reader_vocab: {...},         # domain nouns / synonyms
  trusted: [node_ids or claim_ids]
}
```
`build_H(plan, claims) -> H, y, w`. One function. Domain never appears in the estimator.

### 3e. What to measure once it exists
- Same estimator, two or three plan files, side by side: recovery rate and false certainty per domain. That's the proof of generality.
- Per-plan identifiability: for each plan, enumerate minimal `a = Hc` supports and the resolving measurement. The vulnerability map per domain.
- Recoverability bound per plan: sweep `k`, find the knee, compare to the `2k`-rank condition.

---

## 4. Priorities and non-goals

Do first:
1. Disagreement weighting (replace consensus; `w_i` into weighted L1). One hour. Makes the novelty claim true.
2. Metrics harness that emits JSON: false certainty, state error, recovery rate, detection precision/recall, wall-clock. Run for LS vs L1 across `k`, N seeds.
3. Method freeze commit hash in README; adversarial cases chosen after freeze (correlated single-source, leaf vs hub, coordinated pair).
4. Plan schema stub in the repo, with supply and LeakDB expressed as two plan files.
5. Identifiability enumeration for one plan; table of invisible supports → resolving reading.

Do not build: telephony/phone agent, live "try to fool it" input, UI polish, time dynamics, anything not producing a number or a schema.

---

## 5. Constraints on the estimator (do not violate)
- Estimator never receives the corruption mask or ground truth. Enforce in code, not convention.
- No LLM inside the estimator. Agents read and explain; math decides.
- State assumptions explicitly: sparse corruption, support unknown, magnitude unbounded, coordination allowed. Say where L1's guarantee ends.
- Report false certainty. If the ambiguity set has more than one member, return the set, never a single guess.

---

## 6. Sources to cite
- Candès & Tao, "Decoding by Linear Programming," IEEE Trans. Inf. Theory 2005.
- Fawzi, Tabuada, Diggavi, "Secure Estimation and Control for Cyber-Physical Systems Under Adversarial Attacks," IEEE TAC 2014.
- Liu, Ning, Reiter, "False Data Injection Attacks against State Estimation in Electric Power Grids," ACM TISSEC 2011 (stealthy `a = Hc` attacks).
- Tamhane & Mah, data reconciliation / gross error detection, Technometrics 1985.
- Candès, Wakin, Boyd, "Enhancing Sparsity by Reweighted ℓ1 Minimization," 2008 (optional upgrade).
- LeakDB benchmark (water distribution leak dataset).
