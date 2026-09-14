FUEL SCENARIO SET
=================

Five CSVs on ONE network, differing only in what corruption is planted. Every
expected value below was verified by solving the system independently before
these files were written.

THE NETWORK (different shape from the supply fixtures — this one is a MESH)

    PORT_HAVEN ──┬── FOB_IRONSIDE ──┐
                 │                  ├── OP_TALON
                 └── FOB_KESTREL ───┤
                                    └── OP_VIPER

OP_TALON is fed by BOTH forward bases. That makes this a mesh, not a tree, which
is worth having: a tree gives every node exactly one inflow path, so some bugs
hide. The two-parent node exercises the balance equation properly.

Quantity: fuel gallons. Five transfers over one day, each reported twice (sender
and receiver).

    Site            Opening    True EOD
    PORT_HAVEN       20000       11000
    FOB_IRONSIDE      3000        7100
    FOB_KESTREL       2500        5200
    OP_TALON           400        1900
    OP_VIPER           350        1050

Total conserved at 26250 both ends.

Transfers: PORT->IRONSIDE 5000, PORT->KESTREL 4000, IRONSIDE->TALON 900,
KESTREL->TALON 600, KESTREL->VIPER 700.

All five files report correctable_k = 1.


THE FIVE SCENARIOS
------------------

fuel_C_clean.csv          no corruption at all
    Expect: max error 0.0, zero claims flagged, every node green.
    This is the control. If anything flags here, the threshold is wrong.

fuel_B_two_corruptions.csv    two independent bad claims
    OP_TALON reports EOD 1450 (true 1900)
    IRONSIDE->TALON driver reports 1300 sent (true 900)
    Expect: all five nodes recovered EXACTLY, and exactly two claims flagged:
        eod_OP_TALON                  residual -450
        sent_FOB_IRONSIDE->OP_TALON   residual +400
    Note this recovers two corruptions on a graph whose guarantee is k=1. The
    2k-row condition is a worst-case bound, not a prediction — these two
    corruptions happen to be separable. Worth saying out loud if asked.

fuel_D_correlated.csv     one reporter wrong on everything it touches
    FOB_KESTREL's log cell understates by 120 on all three of its claims.
    Expect: all five nodes recovered exactly, and three claims flagged, all at
    -120:
        eod_FOB_KESTREL
        sent_FOB_KESTREL->OP_TALON
        sent_FOB_KESTREL->OP_VIPER
    This is the correlated-corruption case. The residuals cluster by reporter,
    which is the signal a source-level weighting scheme would use.

fuel_E_consumption.csv    known sinks + one corruption
    Adds a consumption channel: each site burns fuel daily (IRONSIDE 800,
    KESTREL 600, TALON 150, VIPER 120). Set sinks = "known".
    True EOD with burn: PORT 11000, IRONSIDE 6300, KESTREL 4600, TALON 1750,
    VIPER 930.
    OP_VIPER reports 980 (true 930).
    Expect: all five recovered exactly, only eod_OP_VIPER flagged at +50.
    This one matters because the corruption (50) is small relative to the burn
    rates (600-800). It tests that consumption is subtracted rather than
    absorbed into the residual.

fuel_F_overcorrupted.csv  FOUR bad claims on a k=1 graph — the honest limit
    OP_TALON 1450, OP_VIPER 1400, FOB_KESTREL 4850, and the IRONSIDE->TALON
    driver at 1300.
    Expect FAILURE, and specifically this failure:
        FOB_KESTREL off by -350, OP_VIPER off by +350
        four claims flagged, including recv_FOB_KESTREL->OP_VIPER, which is
        HONEST — the estimator misattributes
    This is not a bug. It is what happens past the guarantee, and it is the
    scenario to show when someone asks where the method breaks. The
    identifiability report already said k=1; this demonstrates the consequence
    of exceeding it.


HOW TO USE THESE
----------------

C then B is the demo pair: clean board, then two corruptions caught with the
true state recovered anyway.

D is the correlated case, which is where your sweep showed false certainty
climbing. Here it still recovers, because three claims from one reporter on a
graph with this much redundancy are separable. Useful contrast to the sweep.

F is the slide for honest limits. Run it and say: the system told us k=1 up
front, we gave it four, and here is exactly how it fails.

fuel_truth.json is scoring only. Do not upload it.
