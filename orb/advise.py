"""
orb/advise.py
-------------
Turn compile + decode into an ops playbook: what to do, and where to put sensors.

This is not a second estimator. It only reads the graph, compile report, and
decode output. A distribution lead should be able to act from `actions` without
knowing H, k, or residuals.
"""

from __future__ import annotations

from collections import Counter, defaultdict


_SENT = ("sent", "send", "sending", "ship", "shipment", "dispatch", "load")
_RECV = ("received", "receive", "got", "receipt")
# Planning deadband: MAD can explode on 4-claim snapshots; a crate still matters.
PLAN_DEADBAND = 5.0


def _ops_thresh(decode_thresh: float) -> float:
    t = float(decode_thresh or 0.0)
    if t <= 0:
        return PLAN_DEADBAND
    return min(t, PLAN_DEADBAND)


def generate_advice(graph: dict, compiled: dict, decoded: dict) -> dict:
    """
    Returns
    -------
    headline, severity, actions, sensors, on_hand, hops
    """
    report = (compiled or {}).get("report") or {}
    graph = graph or {}
    decoded = decoded or {}

    k = int(report.get("correctable_k") or decoded.get("correctable_k") or 0)
    thresh = float(decoded.get("threshold") or 0.5)
    ops_t = _ops_thresh(thresh)
    qty_obs = {str(n): int(v) for n, v in (report.get("qty_obs_counts") or {}).items()}
    metered = {str(x) for x in (report.get("metered_sink_ids") or [])}
    blinds = list(decoded.get("undetectable") or report.get("blind_edges") or [])
    loss = list(decoded.get("loss") or [])
    ambiguous = list(decoded.get("ambiguous") or [])
    flagged = list(decoded.get("flagged") or [])
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    claims = list(graph.get("claims") or [])

    degree = Counter()
    for e in edges:
        degree[str(e.get("from"))] += 1
        degree[str(e.get("to"))] += 1

    l1_qty = {n["id"]: float(n["qty"]) for n in decoded.get("nodes") or []}
    l1_flow = {e["id"]: float(e["flow"]) for e in decoded.get("edges") or []}
    claimed_qty = _claimed_node_qty(claims)
    claimed_flow = _claimed_edge_flow(claims)
    matched = _matched_dual_hops(graph, ops_t)
    corrected_hops = []
    for e in edges:
        eid = e.get("id")
        l1 = l1_flow.get(eid)
        claimed = claimed_flow.get(eid)
        if l1 is None or claimed is None:
            continue
        if abs(l1 - claimed) > ops_t:
            corrected_hops.append({
                "id": eid,
                "from": e.get("from"),
                "to": e.get("to"),
                "l1": l1,
                "claimed": claimed,
            })

    sensors = _sensor_plan(
        blinds=blinds,
        ambiguous=ambiguous,
        matched=matched,
        flagged=flagged,
        k=k,
        qty_obs=qty_obs,
        metered=metered,
        degree=degree,
        nodes=nodes,
        edges=edges,
        l1_flow=l1_flow,
        thresh=thresh,
        corrected_hops=corrected_hops,
    )

    actions = _actions(
        graph=graph,
        loss=loss,
        ambiguous=ambiguous,
        flagged=flagged,
        blinds=blinds,
        matched=matched,
        sensors=sensors,
        k=k,
        thresh=thresh,
        l1_qty=l1_qty,
        claimed_qty=claimed_qty,
        corrected_hops=corrected_hops,
        nodes=nodes,
        claims=claims,
    )

    on_hand = _on_hand(nodes, l1_qty, claimed_qty, ops_t, loss, blinds)
    hops = _hop_board(
        edges, l1_flow, matched, blinds, flagged, claims, qty_obs, corrected_hops,
    )

    severity, headline = _headline(
        loss, flagged, blinds, ambiguous, k, sensors, corrected_hops,
    )

    return {
        "headline": headline,
        "severity": severity,
        "correctable_k": k,
        "actions": actions,
        "sensors": sensors,
        "on_hand": on_hand,
        "hops": hops,
    }


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------

def _sensor_plan(
    *,
    blinds,
    ambiguous,
    matched,
    flagged,
    k,
    qty_obs,
    metered,
    degree,
    nodes,
    edges,
    l1_flow,
    thresh,
    corrected_hops,
) -> list[dict]:
    sensors: list[dict] = []
    used_at_kind: set[tuple[str, str]] = set()

    def add(item: dict) -> None:
        key = (item["at"], item["kind"])
        if key in used_at_kind:
            return
        used_at_kind.add(key)
        sensors.append(item)

    # 1. Cover coordinated-lie blind hops: one independent closeout per hop-set.
    uncovered = [dict(b) for b in blinds]
    while uncovered:
        scores: Counter = Counter()
        for b in uncovered:
            for nid in (b.get("from"), b.get("to")):
                if nid:
                    scores[str(nid)] += 1
        if not scores:
            break

        def rank(nid: str) -> tuple:
            incoming = sum(1 for e in edges if str(e.get("to")) == nid)
            return (
                scores[nid],
                -int(qty_obs.get(nid, 0) or 0),
                incoming,
                -int(degree.get(nid, 0) or 0),
                nid,
            )

        best = max(scores, key=rank)
        covered = [
            b for b in uncovered
            if str(b.get("from")) == best or str(b.get("to")) == best
        ]
        add({
            "at": best,
            "kind": "independent_count",
            "required": True,
            "instrument": (
                "Independent EOD / tank-level / warehouse closeout "
                "(not filed by the hop clerks)"
            ),
            "covers_hops": [
                {"id": b.get("id"), "from": b.get("from"), "to": b.get("to")}
                for b in covered
            ],
            "effect": (
                "Third channel on these hops. A matched send/receive lie "
                "of any size will then unbalance this node."
            ),
            "reason": (
                f"{len(covered)} hop(s) have no stock claim on either end "
                "(coordinated-lie blind). L1 cannot see a shared false transfer."
            ),
        })
        covered_ids = {(b.get("id"), b.get("from"), b.get("to")) for b in covered}
        uncovered = [
            b for b in uncovered
            if (b.get("id"), b.get("from"), b.get("to")) not in covered_ids
        ]

    # 2. Ambiguous sinks: drain meter + keep/add a qty sensor.
    for a in sorted(ambiguous, key=lambda x: abs(float(x.get("sink") or 0)), reverse=True):
        nid = str(a.get("id") or "")
        if not nid:
            continue
        has_qty = int(qty_obs.get(nid, 0) or 0) >= 1
        has_meter = nid in metered
        if has_meter and has_qty:
            continue
        kind = "drain_meter" if not has_meter else "second_qty"
        mag = abs(float(a.get("sink") or 0))
        add({
            "at": nid,
            "kind": kind,
            "required": True,
            "instrument": (
                "Drain / consumption meter plus a stock/level sensor"
                if not has_meter else
                "Second independent qty/level sensor (drain meter already present)"
            ),
            "covers_hops": [],
            "effect": (
                "Raises correctable_k. L1 can then name physical loss vs a "
                "downward qty/EOD lie instead of returning the set."
            ),
            "reason": (
                f"Unmetered sink ~{mag:.0f} at {nid}: leak and a low closeout "
                "are the same residual signature."
            ),
        })

    # 3. Conservation holds on dual-reported hops that also have closeouts.
    #    L1 cannot tell truth from a coordinated lie that rewrote EOD too.
    #    This is hardening, not an emergency — every clean logistics net looks
    #    the same from inside the snapshot.
    if not blinds and not flagged and not ambiguous and not corrected_hops:
        both_counted = [
            h for h in matched
            if int(qty_obs.get(str(h.get("from")), 0) or 0) > 0
            and int(qty_obs.get(str(h.get("to")), 0) or 0) > 0
        ]
        if both_counted:
            hop = max(
                both_counted,
                key=lambda h: abs(float(l1_flow.get(h.get("id"), h.get("sent") or 0))),
            )
            frm, to = str(hop.get("from")), str(hop.get("to"))
            at = frm if int(degree.get(frm, 0)) <= int(degree.get(to, 0)) else to
            add({
                "at": at,
                "kind": "independent_count",
                "required": False,
                "instrument": (
                    "Independent physical scale / SCADA level that is not "
                    "the same closeout the hop clerks file"
                ),
                "covers_hops": [{"id": hop.get("id"), "from": frm, "to": to}],
                "effect": (
                    "Breaks a = Hc on this hop: a shared false transfer plus "
                    "matching closeouts will no longer conserve."
                ),
                "reason": (
                    "L1 is silent because the picture conserves. Sender, receiver, "
                    "and closeout can still collude. An independent sensor is the "
                    "only detector for that class of lie."
                ),
            })

    # 4. k == 0 and no specific target yet: instrument the busiest unmetered node.
    if k == 0 and not sensors and nodes:
        candidates = [str(n.get("id")) for n in nodes if n.get("id")]
        if candidates:
            at = max(
                candidates,
                key=lambda nid: (int(degree.get(nid, 0)), -int(qty_obs.get(nid, 0) or 0)),
            )
            add({
                "at": at,
                "kind": "drain_meter",
                "required": True,
                "instrument": "Drain meter plus a stock/level sensor on the same node",
                "covers_hops": [],
                "effect": "Gives L1 a two-sensor site so loss and downward lies split.",
                "reason": (
                    "correctable_k is 0: the estimator cannot tell loss from a "
                    "downward qty lie anywhere on this net."
                ),
            })

    return sensors


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _actions(
    *,
    graph,
    loss,
    ambiguous,
    flagged,
    blinds,
    matched,
    sensors,
    k,
    thresh,
    l1_qty,
    claimed_qty,
    corrected_hops,
    nodes,
    claims,
) -> list[dict]:
    actions: list[dict] = []
    claims_by_id = {c.get("id"): c for c in claims if c.get("id") is not None}

    for s in sorted(loss, key=lambda x: abs(float(x.get("sink") or 0)), reverse=True):
        nid = str(s.get("id"))
        mag = abs(float(s.get("sink") or 0))
        qty = l1_qty.get(nid)
        actions.append({
            "priority": 0,
            "kind": "resupply",
            "title": f"Treat {nid} as short {mag:.0f}",
            "do": (
                f"Do not ship from {nid} until a physical count and restock. "
                f"L1 named a real withdrawal of {mag:.0f} here"
                + (f"; estimated on-hand is {qty:.0f}." if qty is not None else ".")
            ),
            "where": [nid],
            "why": (
                "A drain meter and a qty sensor (or T≥3 with k≥1) split loss "
                "from a downward lie. This is a shortage, not a paperwork error."
            ),
        })

    flagged_sites: list[str] = []
    for f in flagged:
        claim = claims_by_id.get(f.get("claim_id"), {})
        ref = claim.get("ref") or f.get("source") or f.get("claim_id")
        site = _site_for_claim(claim, graph)
        if site:
            flagged_sites.append(site)
        val = claim.get("value")
        actions.append({
            "priority": 0,
            "kind": "recount",
            "title": f"Do not plan on {f.get('claim_id')} ({f.get('source', 'report')})",
            "do": (
                f"Hold any lift that depends on this number. Send a count team "
                f"to {site or ref}. Claimed {val}; residual "
                f"|r|={abs(float(f.get('residual') or 0)):.1f}."
            ),
            "where": [site] if site else [],
            "why": "Sparse residual: this report does not conserve with the rest of the net.",
        })

    for h in corrected_hops:
        frm, to = h.get("from"), h.get("to")
        actions.append({
            "priority": 0,
            "kind": "verify_hop",
            "title": f"Plan on L1, not the radio, for {frm} → {to}",
            "do": (
                f"Hold any allocation that used the reported "
                f"{h.get('claimed'):.0f}. L1 recovered {h.get('l1'):.0f} on this hop. "
                f"Count both ends before the next truck."
            ),
            "where": [x for x in (frm, to) if x],
            "why": (
                "Reported transfer does not match conserved closeouts. "
                "A matched send/receive is not proof when the books disagree."
            ),
        })

    # Hold outbound from a flagged site (one action, not one per claim).
    hold_from = sorted({s for s in flagged_sites if s})
    if hold_from:
        actions.append({
            "priority": 0,
            "kind": "hold",
            "title": "Hold outbound from flagged sites",
            "do": (
                "Do not dispatch the next truck from "
                + ", ".join(hold_from)
                + " until the recount lands. Their on-hand is not a planning number."
            ),
            "where": hold_from,
            "why": "Shipping against a corrected closeout strands the next hop.",
        })

    for a in sorted(ambiguous, key=lambda x: abs(float(x.get("sink") or 0)), reverse=True):
        nid = str(a.get("id"))
        mag = abs(float(a.get("sink") or 0))
        actions.append({
            "priority": 1,
            "kind": "do_not_treat_as_leak",
            "title": f"Do not treat {nid} as a confirmed leak",
            "do": (
                f"Physical count at {nid} now. The missing {mag:.0f} is either "
                f"loss or a low closeout — L1 cannot tell. Install a drain meter "
                f"before the next snapshot."
            ),
            "where": [nid],
            "why": a.get("reason") or "Unmetered unknown sink: leak and downward corruption share a signature.",
        })

    for b in blinds:
        frm, to = b.get("from"), b.get("to")
        cover = next(
            (
                s.get("at")
                for s in sensors
                if any(
                    str(h.get("from")) == str(frm) and str(h.get("to")) == str(to)
                    for h in (s.get("covers_hops") or [])
                )
            ),
            None,
        )
        place = cover or to or frm
        actions.append({
            "priority": 1,
            "kind": "verify_hop",
            "title": f"Do not treat {frm} → {to} as verified",
            "do": (
                f"Both ends only cross-confirmed the same number. Place an "
                f"independent closeout at {place} before you plan on this lift."
            ),
            "where": [x for x in (frm, to) if x],
            "why": b.get("reason") or "No third channel; a coordinated sender/receiver lie is undetectable.",
        })

    for s in sensors:
        if s["kind"] == "independent_count" and any(
            a.get("kind") == "verify_hop" and s["at"] in (a.get("where") or [])
            for a in actions
        ):
            # Placement is already implied by the verify_hop action; still surface it
            # as a dedicated sensor action so the dashboard has a placement card.
            pass
        hops = s.get("covers_hops") or []
        hop_txt = ""
        if hops:
            hop_txt = " Covers " + "; ".join(
                f"{h.get('from')}→{h.get('to')}" for h in hops
            ) + "."
        actions.append({
            "priority": 1 if s.get("required", True) else 3,
            "kind": "place_sensor",
            "title": f"Place { _sensor_short(s['kind']) } at {s['at']}",
            "do": f"{s['instrument']} at {s['at']}.{hop_txt} {s['effect']}",
            "where": [s["at"]],
            "why": s["reason"],
        })

    # Low on-hand (L1), excluding named-loss sites and endpoints of blind hops.
    # Qty implied only by an unverified transfer is not a planning number.
    loss_ids = {str(s.get("id")) for s in loss}
    unverified = _blind_nodes(blinds)
    resupply = []
    edges = graph.get("edges") or []
    for n in nodes:
        nid = str(n.get("id") or "")
        if not nid or nid in loss_ids or nid in unverified:
            continue
        qty = l1_qty.get(nid)
        if qty is None:
            continue
        initial = float(n.get("initial") or 0)
        low = qty < 0 or (initial > 0 and qty < 0.2 * initial) or qty < 50
        if not low:
            continue
        incoming = any(e.get("to") == nid for e in edges)
        if not incoming and qty >= 0:
            continue
        resupply.append((qty, nid, initial))
    resupply.sort()
    for qty, nid, initial in resupply[:4]:
        actions.append({
            "priority": 2,
            "kind": "resupply",
            "title": f"{nid} is low on L1 books ({qty:.0f})",
            "do": (
                f"Queue a lift into {nid}. Estimated on-hand {qty:.0f}"
                + (f" vs opening {initial:.0f}." if initial else ".")
            ),
            "where": [nid],
            "why": "Use L1 on-hand, not the last radio report, for the next allocation.",
        })

    if k == 0 and not loss and ambiguous:
        actions.append({
            "priority": 2,
            "kind": "cannot_certify",
            "title": "Do not certify this picture as clean",
            "do": (
                "correctable_k is 0. L1 cannot tell a leak from a downward lie. "
                "Treat every shortage as a set {loss, false closeout} until a "
                "second sensor is in."
            ),
            "where": [],
            "why": "One sensor type at a node leaves a null space of size 1.",
        })

    if not actions:
        actions.append({
            "priority": 3,
            "kind": "clear",
            "title": "No sparse corruption in this snapshot",
            "do": (
                "Books conserve. You can plan on L1 on-hand for this lift. "
                "A matched send/receive is still not independent verification — "
                "place one closeout on any hop that has none."
            ),
            "where": [],
            "why": "Residuals are in-band and identifiability is not blocking.",
        })

    # De-dupe place_sensor that repeats a verify_hop sentence; keep both if
    # the sensor is the actual instruction. Sort: priority, then kind.
    actions.sort(key=lambda a: (int(a.get("priority", 9)), str(a.get("kind")), str(a.get("title"))))
    return actions


def _sensor_short(kind: str) -> str:
    return {
        "independent_count": "an independent closeout",
        "drain_meter": "a drain meter",
        "second_qty": "a second qty sensor",
    }.get(kind, "a sensor")


def _site_for_claim(claim: dict, graph: dict) -> str | None:
    if not claim:
        return None
    if claim.get("type") == "node" or claim.get("type") == "sink":
        ref = claim.get("ref")
        return str(ref) if ref else None
    if claim.get("type") == "edge":
        eid = claim.get("ref")
        for e in graph.get("edges") or []:
            if e.get("id") == eid:
                src = str(claim.get("source") or "").lower()
                if any(t in src for t in _RECV):
                    return str(e.get("to"))
                return str(e.get("from"))
    return None


# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------

def _blind_nodes(blinds) -> set[str]:
    out: set[str] = set()
    for b in blinds or []:
        if b.get("from"):
            out.add(str(b["from"]))
        if b.get("to"):
            out.add(str(b["to"]))
    return out


def _on_hand(nodes, l1_qty, claimed_qty, thresh, loss, blinds) -> list[dict]:
    loss_ids = {str(s.get("id")): float(s.get("sink") or 0) for s in loss}
    unverified = _blind_nodes(blinds)
    rows = []
    for n in nodes:
        nid = str(n.get("id") or "")
        if not nid:
            continue
        l1 = l1_qty.get(nid)
        reported = claimed_qty.get(nid)
        delta = None
        if l1 is not None and reported is not None:
            delta = abs(l1 - reported)
        action = "plan on L1"
        if nid in unverified:
            action = "unverified hop — do not plan on this qty"
        elif nid in loss_ids:
            action = "resupply — named shortage"
        elif delta is not None and delta > thresh:
            action = "recount — reported ≠ L1"
        elif reported is None and l1 is not None:
            action = "no closeout on file"
        rows.append({
            "id": nid,
            "opening": float(n.get("initial") or 0),
            "reported": reported,
            "l1": l1,
            "delta": delta,
            "action": action,
        })
    rows.sort(key=lambda r: (r["l1"] is None, r["l1"] if r["l1"] is not None else 0))
    return rows


def _hop_board(edges, l1_flow, matched, blinds, flagged, claims, qty_obs, corrected_hops) -> dict:
    blind_ids = {b.get("id") for b in blinds}
    matched_ids = {h.get("id") for h in matched}
    corrected_ids = {h.get("id") for h in corrected_hops}
    flagged_edge_claims = [f for f in flagged if str(f.get("type")) == "edge"]
    trust, audit = [], []
    for e in edges:
        eid = e.get("id")
        row = {
            "id": eid,
            "from": e.get("from"),
            "to": e.get("to"),
            "l1_flow": l1_flow.get(eid),
        }
        if eid in corrected_ids:
            h = next((x for x in corrected_hops if x.get("id") == eid), {})
            row["tag"] = (
                f"L1 {h.get('l1'):.0f} vs reported {h.get('claimed'):.0f} — do not plan on the radio"
            )
            audit.append(row)
        elif eid in blind_ids:
            row["tag"] = "unverified — no third channel"
            audit.append(row)
        elif any(
            (claims_ref(claims, f.get("claim_id")) == eid)
            for f in flagged_edge_claims
        ):
            row["tag"] = "flagged — do not plan on this lift"
            audit.append(row)
        elif eid in matched_ids:
            hop = next((h for h in matched if h.get("id") == eid), {})
            frm, to = str(hop.get("from") or e.get("from")), str(hop.get("to") or e.get("to"))
            has_from = int(qty_obs.get(frm, 0) or 0) > 0
            has_to = int(qty_obs.get(to, 0) or 0) > 0
            row["sent"] = hop.get("sent")
            row["received"] = hop.get("received")
            if not has_from and not has_to:
                row["tag"] = "matched send/receive — not independent"
                audit.append(row)
            else:
                row["tag"] = "send/receive + closeout agree"
                trust.append(row)
        else:
            row["tag"] = "consistent"
            trust.append(row)
    return {"trust": trust, "audit": audit}


def claims_ref(claims, claim_id) -> str | None:
    for c in claims:
        if c.get("id") == claim_id:
            return c.get("ref")
    return None


def _headline(loss, flagged, blinds, ambiguous, k, sensors, corrected_hops) -> tuple[str, str]:
    if loss:
        s = loss[0]
        return "critical", (
            f"Named shortage at {s.get('id')} ({abs(float(s.get('sink') or 0)):.0f}). "
            "Resupply; do not ship from there."
        )
    if corrected_hops:
        h = corrected_hops[0]
        return "critical", (
            f"L1 recovered {h.get('l1'):.0f} on {h.get('from')} → {h.get('to')} "
            f"(radio said {h.get('claimed'):.0f}). Do not plan on the reported lift."
        )
    if flagged:
        return "critical", (
            f"{len(flagged)} report(s) fail conservation. Recount before the next lift."
        )
    if blinds:
        return "watch", (
            f"{len(blinds)} hop(s) have no independent count. "
            "Do not treat those transfers as verified — place a closeout."
        )
    if ambiguous:
        a = ambiguous[0]
        return "watch", (
            f"Possible leak or false closeout at {a.get('id')} "
            f"({abs(float(a.get('sink') or 0)):.0f}). Count it; do not assume a leak."
        )
    if k == 0:
        return "watch", (
            "L1 cannot certify this picture (correctable_k = 0). "
            "Add a drain meter and a stock sensor before you treat shortages as real."
        )
    if any(s.get("kind") == "independent_count" and s.get("required") for s in sensors):
        return "watch", (
            "Books conserve, but a coordinated lie can still hide. "
            "Place an independent sensor on the dual-reported hop."
        )
    return "clear", "Books conserve. Plan on L1 on-hand for this snapshot."


# ---------------------------------------------------------------------------
# Claim helpers
# ---------------------------------------------------------------------------

def _claimed_node_qty(claims) -> dict[str, float]:
    sums: dict[str, list[float]] = defaultdict(list)
    skip = ("consum", "used", "usage", "issued", "burned", "withdraw")
    for c in claims:
        if c.get("type") != "node" or c.get("ref") is None:
            continue
        src = str(c.get("source") or "").lower()
        if any(tok in src for tok in skip):
            continue
        try:
            v = float(c.get("value"))
        except (TypeError, ValueError):
            continue
        sums[str(c["ref"])].append(v)
    return {k: sum(vs) / len(vs) for k, vs in sums.items() if vs}


def _claimed_edge_flow(claims) -> dict[str, float]:
    sums: dict[str, list[float]] = defaultdict(list)
    for c in claims:
        if c.get("type") != "edge" or c.get("ref") is None:
            continue
        try:
            v = float(c.get("value"))
        except (TypeError, ValueError):
            continue
        sums[str(c["ref"])].append(v)
    return {k: sum(vs) / len(vs) for k, vs in sums.items() if vs}


def _matched_dual_hops(graph: dict, thresh: float) -> list[dict]:
    by_edge: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"sent": [], "recv": []})
    for c in graph.get("claims") or []:
        if c.get("type") != "edge" or not c.get("ref"):
            continue
        src = str(c.get("source") or "").lower()
        try:
            v = float(c.get("value"))
        except (TypeError, ValueError):
            continue
        eid = str(c["ref"])
        if any(t in src for t in _RECV):
            by_edge[eid]["recv"].append(v)
        elif any(t in src for t in _SENT):
            by_edge[eid]["sent"].append(v)
    id_to_e = {e.get("id"): e for e in graph.get("edges") or []}
    out = []
    for eid, bags in by_edge.items():
        if not bags["sent"] or not bags["recv"]:
            continue
        sent = sum(bags["sent"]) / len(bags["sent"])
        recv = sum(bags["recv"]) / len(bags["recv"])
        if abs(sent - recv) > thresh:
            continue
        e = id_to_e.get(eid) or {}
        out.append({
            "id": eid,
            "from": e.get("from"),
            "to": e.get("to"),
            "sent": sent,
            "received": recv,
        })
    return out
