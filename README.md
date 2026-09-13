# orb
# ORB — Project Context Handoff

**Event:** East v. West 72-Hour Hackathon, Sept 12–14 2026
**Track:** Defense — "Resilient estimation for critical infrastructure"
**Team:** Two full-stack engineers. Strong at agents and AI apps. No prior background in estimation, control theory, or power systems. Using Claude Fable, GPT Astra, and Claude Code heavily.
**Track position:** One of only two teams on this track. Judges will have time to read closely. Depth and honesty beat flash.

---

## 1. What the challenge is asking, in plain language

Some of your sensors are lying and you don't know which ones. Figure out what's actually true anyway. If you can't, say so instead of guessing.

The brief frames this for critical infrastructure: power grids, water systems, emergency supply chains. A decision-maker receives reports about the state of a system. Some reports are delayed, missing, or wrong. Several can be wrong at once, and wrong reports can be coordinated. The decision-maker needs the best estimate of the true state AND a clear signal of whether that estimate is actually supported by the surviving evidence.

The brief lists five acceptable contributions:
- Build a robust estimator
- Compare recovery strategies
- Derive a recoverability bound (how many bad reports can be survived)
- Optimize sensor placement
- Construct a simulation that exposes a hidden failure mode

You do not need all five. Two done properly is a complete project. A rigorous negative result (showing precisely where a method fails and why) counts as a real contribution.

Key rules from the brief:
- The estimator must NEVER be told which sensors are corrupted. It has to infer from the data.
- Test cases must be chosen AFTER the method is fixed. No tuning against your test set.
- A polished interface does not establish a claim. Evidence does.
- State your assumptions and limitations explicitly.
- Report false certainty: how often the system outputs a confident wrong answer. This is the headline metric.

Checkpoints (Eastern Time): Sat 10pm, Sun 10am, Sun 10pm, Mon 10am, final Mon noon. Each checkpoint: 60-second video + GitHub repo. Missing a checkpoint = zero for that block. Checkpoints accept incomplete work; explain what changed and where feedback would help.

Event-wide rubric: Innovation 30%, Technical 25%, Business Value 25%, Presentation 20%.

---

## 2. Core concepts (learned from zero — these are the only ones you need)

### Sensors, readings, hidden state
There is a hidden true state you cannot observe directly. Sensors each report ONE number. That one number can depend on SEVERAL hidden values at once. Example: a counter in a hallway between two rooms reports one number, but that number is the sum of two room headcounts.

### The equation: y = Hx + a + ε
- `x` = hidden true state (what you want)
- `H` = the table saying "if the truth is x, sensor i should read this" — one row per sensor
- `ε` = ordinary small random noise
- `a` = corruption vector: mostly zeros, but wherever a sensor is lying, that entry can be anything
- `y` = what you actually receive

### Noise vs corruption
Noise is small, random, averages out. Standard methods handle it.
Corruption is arbitrarily large, can be biased, can be coordinated across sensors. Standard methods do NOT handle it — they absorb the lie into the answer.

### Residual
Take a candidate answer. Compute what each sensor SHOULD read if that answer were true. Subtract what it DID read. That difference is the residual. Small residuals everywhere = the story holds. Large residual somewhere = contradiction. Every method in this project runs on residuals.

### Redundancy (spare sensors)
With exactly as many sensors as unknowns, every reading is load-bearing. A lie just becomes the answer and nothing looks wrong. Spare sensors create contradictions. Roughly: one spare lets you NOTICE a liar exists, two spares let you IDENTIFY which one.

### Least squares (the compromiser)
Finds the answer that minimizes the sum of SQUARED residuals. Assumes everyone is honest but slightly imprecise. Splits the difference across all sensors. Against a single big lie, it smears the error across everyone — and the largest residual often lands on an INNOCENT sensor. Returns one confident answer, always, even when the information isn't there. This is the baseline that fails.

Worked example: three rooms a,b,c with truth 10,20,30. Four sensors: a+b, b+c, a+c, a+b+c. Sensor 2 lies and reports 99 instead of 50. Least squares returns a=−18, b=41, c=51. Residuals are 7, 7, 7, −14 — the biggest is on sensor 4, the honest one.

### Subset search (the detective)
Assume at most k sensors lie. For every possible set of k sensors: throw them out, solve with the rest, check whether the kept sensors agree with the result. Keep only the stories that hold together. Returns:
- ONE surviving story → you have the truth and the liar's identity
- SEVERAL surviving stories → genuine ambiguity. Return the SET. Nothing in the data distinguishes them.
- NONE → more liars than assumed. Say so.
This is textbook (1970s). It's the DEFINITION of the problem, not a novel solution. It's exponential — dies on real networks. Use it as ground truth on small cases only.

### L1 minimization (the practical estimator)
Same as least squares but minimizes the sum of ABSOLUTE residuals instead of squared. Squaring punishes big errors brutally, so least squares would rather have four medium errors than one huge one — that's why it smears. L1 is nearly indifferent, so it isolates the outlier instead of absorbing it. L1 solutions are SPARSE: most residuals zero, a few large. That matches "few sensors lie." It's a linear program — solves in milliseconds on thousands of sensors. The secure estimation paper in the brief proves that with enough redundancy, L1 gives the SAME answer as exhaustive subset search. Implementation: `scipy.optimize.linprog` or `cvxpy`. Has a limit: past a certain number of liars relative to redundancy, it starts compromising like least squares.

### Identifiability / invisible attacks
If two different truths, each with a different set of liars, produce IDENTICAL readings, no algorithm can tell them apart. Not a compute problem — an information problem. Formally: a corruption `a` is undetectable when `a = Hc` for some state shift `c`, because it looks exactly like a legitimate state change. This is why sensor PLACEMENT matters as much as count.

### Why some sensors are worse to corrupt than others
Two separate reasons:
1. **Corroboration.** A sensor with no other sensor checking it can lie freely. Extreme case: a "critical measurement" whose removal makes the system unsolvable — its residual is ALWAYS zero, it can never be caught. Short of that, high-"leverage" sensors bend the estimate toward themselves and mask their own error.
2. **Attack cost.** To silently shift the estimate of node X, an attacker must corrupt every sensor that would notice. A leaf node with two sensors is cheap. A hub with nine is expensive. The sparsity of `Hc` is a per-node vulnerability score.

### Power grid version (IEEE 14-bus) — considered and set aside
Standard test grid: 14 buses (nodes), 20 lines. Hidden state = voltage angle at each bus (13 unknowns; bus 1 is reference). Water analogy: buses are tanks, angle is water level, lines are pipes, flow depends on level difference × pipe width, injection meters measure net hose-minus-drain at a tank, line flow meters measure flow through a pipe. `pandapower` ships `case14`. This is clean numeric data — NO messiness — so an LLM reader has no job. That's why we pivoted.

---

## 3. The project: ORB

### The scenario
Emergency supply logistics. Think base camps: a main depot receives supplies from outside, trucks carry them to forward bases, bases send smaller loads to outposts. The command center cannot see any of it. It only receives messages from people on the ground: text, radio, email. Messages are messy, delayed, duplicated, sometimes wrong, sometimes contradictory.

The brief lists "reconciling inconsistent emergency supply reports" as one of its own examples. This is not a stretch.

### The mapping to the math
- **Tanks / nodes** = bases (depot, forward bases, outposts). Hidden state = true stock at each.
- **Pipes / edges** = supply routes between bases.
- **Water** = ONE supply type. Water cases. Do not track multiple supply types.
- **Flow** = trucks moving supplies along a route.
- **Sensors** = PEOPLE sending reports. A base commander reporting stock = a gauge on a tank. A truck driver reporting a delivery = a gauge on a pipe. A transfer report tells you about TWO hidden quantities (sender went down, receiver went up), exactly like a pipe gauge depending on two tank levels.
- **The conservation rule** = supplies don't appear or vanish. A transfer subtracts from sender and adds to receiver. This rule is what makes lies catchable.
- **Corrupted sensor** = a person whose reports are wrong: guessed, exaggerated to get priority, reported stale numbers, double-reported the same truck, has a broken counting method.
- **Correlated corruption** = one convoy commander runs three trucks and his counting is bad, so all three sites' reports are off together.
- **Extra sensor not present in the grid version** = the LLM reader agent. Before a human message becomes a number, an LLM reads it. If it misreads ("sent 50" → "received 50"), that's corruption injected BEFORE the estimator runs. The reader is a sensor that can also lie.

### Why counting on receipt doesn't solve it
A count on receipt is just another sensor. It can be wrong too. Now you have two reports (driver: 50, receiver: 30) that disagree and both were honest counts. Which is right? That's the ambiguity. In real disasters counting often doesn't happen at all. BUT a careful count is a "trusted reading" — the brief says trusted references resolve ambiguity. So: which bases get careful counters? That's the sensor placement question in this world.

### Architecture: three parts

**Part 1 — The Reader (agent).** Reads messy messages, outputs structured claims. This is where the team's skills are the point.

**Part 2 — The Checker (deterministic math, NO LLM).** Takes claims, builds H, runs L1 (and least squares and subset search as comparisons), returns estimated stock per site + which claims didn't fit. Outputs one of three: a single state, a SET of candidate states with what they disagree on, or "cannot recover."

**Part 3 — The Advisor (agent).** Takes the checker's output and turns it into a decision. "Two states are consistent. They disagree on whether Post 3's delivery arrived. Under both, Base A is safe through tomorrow. One call to Post 3's site lead resolves it. Don't move the truck yet." Can only recommend actions safe across ALL candidate states. Optional but makes the output usable.

One line: **agent builds the grid from messy text → L1 checks it → agent explains the result.**

### The Reader harness (must be more than one prompt)
One prompt per message fails on anything requiring two messages: duplicates, name resolution ("Shelter 4" / "the school on Main" / "site 4"), matching a "sent" to a "received," and confidence ("maybe 30, didn't count" vs "counted twice").

Design:
- **Pass 1 — Extract.** One message at a time, pull raw claims. Cheap, parallel.
- **Pass 2 — Reconcile.** Across all claims: merge duplicates, resolve names to sites, pair transfers with arrivals.
- **Pass 3 — Verify.** For each final claim, show the agent the claim + original message: "does this message actually say this?" Catches sent/received flips.
- **Confidence scoring.** Every claim gets a score. Low-confidence claims become low-WEIGHT rows in L1 — cheap to throw out. Human uncertainty flows into the math.
- **Redundant readers.** Run 3 readers (different prompts, or Fable + Astra + a third). Where they agree, high confidence. Where they disagree, that disagreement IS a residual — the same trick as three gauges on one pipe. Vote or flag. Then build ONE clean claim list, run L1 once. Readers must be genuinely different (different prompts/models); three identical readers make identical mistakes.

What NOT to build: agent swarms, planning loops, tool calling, memory systems. None earn points here.

What earns points: MEASURE it. One reader vs three-pass vs three readers with voting, on the same messages with the same planted lies. Report claim accuracy for each. Then orchestration is a result with a number, not a design choice.

### Data structures (no database)
The "grid" is two small lists in memory. Save as JSON if you want persistence.

```python
sites = ["DEPOT", "BASE_A", "BASE_B", "BASE_C", "POST_1", ..., "POST_6"]

claims = [
  {"type": "stock",    "site": "BASE_A", "amount": 200, "source": "msg_01", "confidence": 0.9},
  {"type": "transfer", "from": "BASE_A", "to": "POST_1", "amount": 50, "source": "msg_01", "confidence": 0.9},
  {"type": "stock",    "site": "POST_1", "amount": 0,   "source": "msg_02", "confidence": 0.5},
]
```

Each claim → one row of H. Stock claim: a 1 under that site. Transfer claim: constraint on the difference between two sites' final stocks (treat starting stocks as known from the depot manifest; document the exact formulation).

Stack: Python, numpy, scipy (`linprog`), Fable/Astra for reader + advisor, React or Streamlit for the UI. If you find yourself setting up Postgres, stop.

---

## 4. What's old vs what's yours (say this out loud in the pitch)

**Old — do not claim as novel:**
- L1 for bad-data detection in state estimation: 1970s, power systems.
- Secure estimation under adversarial attacks: the paper in the brief, ~2014.
- Conservation-based reconciliation: chemical plants have done "data reconciliation" with "gross error detection" for 50 years. Use that phrase.
- LLMs extracting structured facts from text: everyone.

**Likely yours (spend 10 minutes searching to confirm — try "LLM extraction data reconciliation" and "humanitarian logistics report reconciliation"):**
- Combining them on humanitarian/defense logistics where the sensors are people.
- Treating the LLM reader as a fallible sensor inside the failure model.
- Redundant readers with disagreement-as-residual.
- The measurement: how much does reader redundancy improve recovery.

**Always yours regardless:** the evidence. Nobody has run your experiment on your base tree with your planted lies.

The pitch sentence: *"L1 is standard, conservation checking is standard. Our contribution is treating the reader as a fallible sensor and measuring how much redundant readers help."*

---

## 5. Failure model (define this explicitly — the brief demands it)

Report corruption types:
- **Stale** — reports yesterday's number as today's
- **Duplicate** — two people report the same truck
- **Fabricated** — site inflates need to get priority
- **Noise** — honest rounding, ±2
- **Correlated** — one team's counting method is broken, all their sites off together
- **Reader error** — LLM misreads a message (sent/received flip, wrong site, wrong time)
- **Coordinated** — driver AND receiver both lie with matching numbers. Conservation sees nothing wrong. THIS IS THE HIDDEN FAILURE MODE. L1 confidently returns a false picture. Showing this is a result the brief explicitly wants.

State: how many can fail at once (k), whether they're a fixed or changing subset, whether they can coordinate, what stays trusted (e.g. depot manifest, careful-count sites).

---

## 6. Evaluation plan

You invent the truth. Then you invent messages describing it. Then you plant lies. Then you check if the system found the truth.

1. Simulate ground truth: base tree, starting stocks, ~12 transfers over a day, final stocks. Seeded.
2. Generate structured claims from ground truth with redundancy (one stock claim per site, driver + receiver claim per transfer). Add noise. Corrupt k claims. Record which — the estimator NEVER sees this list.
3. LATER: generate messy text messages from ground truth via LLM, with corruption planted at known rates by type.
4. Run pipeline. Measure per configuration:
   - **Stock error** (max and mean vs truth)
   - **Caught the corrupted claims?** (yes/no, and false flags on honest claims)
   - **Ambiguity set correctly sized?** (does it contain the truth? how big is it?)
   - **False certainty rate** — how often it output a single confident answer that was wrong. HEADLINE METRIC.
   - Time to compute
5. Sweep k upward until L1 stops recovering. That number is your recoverability bound for this tree.
6. Move careful-count sites around. Show a counter at BASE_A protects three sites; one at POST_3 protects one.
7. AFTER the method is fixed: pick new corruption cases and see what breaks. Report it.

Snapshot only. One moment in time. The brief says extend to time only after stating what temporal info adds. If snapshot works by Sunday morning, a "day 2" version can watch for stories that don't hold up over a sequence (a site reporting 200, 200, 200 while trucks supposedly deliver). Upside, not required.

"Live" demo = pre-written message stream played in one at a time. Judges know it's simulated.

---

## 7. Which contributions to pick

Recommended two: **compare recovery strategies** (least squares vs L1 vs subset search — shows the method works) and **expose a hidden failure mode** (coordinated lies are invisible — shows you know where it doesn't). Recoverability bound and placement come nearly free once the sweep harness exists.

---

## 8. Build order (step by step, not one-shot)

**Step 1 — World.** `world.py`: 1 depot, 3 bases, 6 outposts as a tree. One supply type. `make_world(seed)` assigns starting stocks, generates ~12 transfers along tree edges (10–80 cases, sender can't exceed stock), applies them, returns sites/edges/start_stock/transfers/final_stock. `print_world()`. Assert total stock conserved → PASS/FAIL. **Check by eye:** totals match, nothing negative, transfers look like a real supply run.

**Step 2 — Structured claims.** `claims.py`: generate stock + transfer claims from truth with redundancy, add ±2 noise, corrupt k claims with wildly wrong numbers, record ground-truth corruption list separately. Seeded, k as parameter.

**Step 3 — Checker.** `estimator.py`: build H from claims (comment the construction clearly). `least_squares()` via `numpy.linalg.lstsq`. `l1_estimate()` via `scipy.optimize.linprog` minimizing sum of absolute residuals. `subset_search()` for small k as ground truth. `flag_claims(residuals, threshold)`. Each returns estimated stocks + residual per claim.

**Step 4 — Test harness.** `test_orb.py`: seed 0, k = 1, 2, 3. Run all estimators. Table: k, estimator, max stock error, caught corrupted (y/n), false flags. **The check:** least squares wrong at k=1,2 while L1 right = on track. THIS IS CHECKPOINT 1 (Sat 10pm).

**Step 5 — Message generator.** LLM writes messy realistic text messages from ground truth. Plant corruption by type at known rates. Save as a fixed dataset (don't regenerate every run — it must be a stable test set).

**Step 6 — Reader v1.** Single prompt, one message at a time → claims JSON. Measure claim accuracy vs the structured ground truth from Step 2.

**Step 7 — Reader harness.** Three-pass (extract / reconcile / verify), confidence scores, redundant readers with voting. Measure again. Report the improvement.

**Step 8 — Weighted L1.** Confidence → row weights. Rerun.

**Step 9 — Sweeps.** k sweep for the bound. Placement sweep for careful-count sites. Coordinated-lie case for the hidden failure.

**Step 10 — Advisor.** Agent over the ambiguity set: what the candidates disagree on, what's safe under all, what single call resolves it.

**Step 11 — UI.** Message stream on left, reconciled state with flagged contradictions in middle, decision panel on right, failure-injection controls on top, evidence tab with the false-certainty chart. Only after Steps 1–9 work headless.

**Step 12 — Post-fix adversarial tests.** Choose new corruption cases after the method is locked. Report what breaks.

Keep everything in a notebook/scripts until the core works. Do NOT build the UI first.

---

## 9. Demo script (60 seconds, three scenes)

1. **Clean.** All reports consistent, both methods agree, green board. Five seconds. Boring on purpose.
2. **Two lies.** Corrupt two reports. Least squares shifts its answer and stays confident — put its wrong number next to the true number. L1 flags the two liars and recovers the truth.
3. **Coordinated lie.** Driver and receiver agree on a false delivery. Least squares: clean, confident, wrong. ORB: "two states are consistent with the reports, here's where they differ, here's the one call that would separate them, here's what's safe meanwhile."

Most projects demo the thing working. ORB demos the thing knowing its own limits.

---

## 10. Risks and the pivot rule

- **Inventing the world takes longer than you think.** Budget four hours, not one.
- **The reader will misread sometimes.** That's fine — it's part of the result. Measure it and say "the reader is a sensor that can also be wrong."
- **Building the UI before the estimator works** scores badly on 25% of the rubric and judges see through the rest.
- **Adding time dynamics before snapshot works.** Don't.
- **Claiming L1 or conservation as novel.** Don't.
- **Leaking ground truth to the estimator** (e.g. rewarding "did you catch the corrupted one"). The estimator only ever sees readings.

**Pivot rule, set now while clear-headed:** if by ~hour 12 least squares isn't failing and L1 isn't recovering on structured claims (Step 4), pivot to a freestyle agent project. Steps 1–4 are ~150 lines total.

---

## 11. Business framing (25% of rubric)

Grid operators, water utilities, emergency logistics, and military sustainment all make decisions from reports that can be wrong, and none currently distinguish "I'm confident" from "I'm guessing." Anyone from the defense side recognizes the problem — it's the same in sensor fusion and every intelligence product with a confidence line. Civilian twins: disaster response, supply chain risk, humanitarian logistics.

---

## 12. Things considered and rejected

- **OSINT analyst agent** — real project, but 200 teams ship LLM wrappers on other tracks; two-team track advantage is worth more.
- **Dark vessel detection (SAR + AIS)** — needs CV/geospatial experience the team lacks; 30 hours fighting rasterio.
- **Pure IEEE 14-bus grid version** — clean data, no role for agents; team would compete on math learned two days ago.
- **Two-agent adversarial RL (corruptor vs fixer)** — valid research shape, and this environment is unusually RL-friendly (matrix ops, microseconds/episode, dense reward). But simultaneous two-agent learning is the most unstable setup in RL; if ever revisited, freeze one and train the other (alternating best response), never let the fixer see the corruption mask, and compare the learned attacker against the analytical null-space attacker or a judge will ask why RL was needed.
- **Any database.** Overkill.
