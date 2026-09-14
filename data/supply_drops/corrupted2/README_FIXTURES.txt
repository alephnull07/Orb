# Supply fixtures

One scenario, three input formats. Same underlying network and same planted corruption, so all three should produce the same answer through different ingest paths.

## The network

Water cases moving through a two-tier distribution tree over one day.

```
MAIN_DEPOT
  ├── FOB_ALPHA ──┬── OP_CRESCENT
  │               └── OP_DELTA
  └── FOB_BRAVO ───── OP_ECHO
```

| Site | Opening | True EOD |
|---|---|---|
| MAIN_DEPOT | 4000 | 2300 |
| FOB_ALPHA | 600 | 1300 |
| FOB_BRAVO | 450 | 900 |
| OP_CRESCENT | 120 | 340 |
| OP_DELTA | 90 | **270** |
| OP_ECHO | 75 | 225 |

Total conserved at 5335 both ends. Six transfers, each reported twice (sender and receiver), which is what creates the redundancy.

## Planted corruption

**OP_DELTA reports EOD on-hand of 520. The true value is 270.** Delta opened at 90 and received a single 180-case delivery, so 270 is pinned by two independent claims plus the balance equation. The 520 claim contradicts them, and its residual is 250.

Nothing else is corrupted.

## Verified

Solved by hand against the same formulation the compiler uses:

- unknowns 11, claims 16, balance rows 6, rank 11 (full)
- **correctable_k = 1**
- L1 recovers all six site values exactly
- exactly one claim flagged: `eod_OP_DELTA`, residual 250

## Files

**`supply_field_reports.txt`** — RECORD mode. 25 bracketed-timestamp radio and text messages. Exercises:
- entity resolution: each site appears under several names (`MAIN_DEPOT` as "the depot", "Depot-Main", "MAIN"; `FOB_ALPHA` as "Alpha", "Alpha-1", "FOB Alpha")
- reader disagreement: one record gives a quantity vaguely ("about 300 cases dropped, did not get an exact recount") and one has no extractable quantity at all ("moved the rest of the damaged pallets off to the side"). Those should come out with low weight, or be dropped.
- redundancy: every transfer has a driver report and a receiver report

**`supply_logistics.csv`** — TABULAR mode. 24 rows, columns `entity_id, channel, value, timestamp, from_node, to_node`. Channels are `opening_on_hand` and `eod_on_hand` (node readings) and `shipment_sent` / `shipment_received` (edge readings). Tests the column-mapping agent: it has to map `channel` values onto node vs edge primitives and read `from_node`/`to_node` for edges.

**`supply_reports.jsonl`** — uniform keys, so the peek should route it to TABULAR. Same 24 records with renamed fields (`site`, `metric`, `qty`, `ts`, `src`, `dst`) plus two distractor columns (`unit`, `reporter`). Tests that the mapping agent isn't keyed to specific column names.

**`supply_truth.json`** — scoring only. The estimator must never read this. Contains opening and true EOD per site, the transfer list, and the planted corruption with its delta.

## Notes

- No field in any input file names the corrupted claim, and no channel or column encodes an answer. The leakage guard should report zero exclusions.
- `sinks` should be `"none"` for all three — supplies are strictly conserved here, nothing leaks or is consumed.
- Opening counts are present as claims, so `known.initial` can either come from them or be left for the estimator to solve. Both work; the numbers above assume openings are trusted.
