#==================================================================
# TESTS FOR INSTRUMENTATION (window-lifecycle latency breakdown)
#==================================================================
from qecsim.config import us
from qecsim.decoders import PresetLatencyDecoder
from qecsim.frontends.circuit import three_cnot_circuit
from qecsim.metrics import WindowLatencyBreakdown
from qecsim.wiring import build_and_run


def test_window_latency_breakdown_stages():
    r = build_and_run(three_cnot_circuit(), num_units=1, d=3, rounds_per_op=11,
                      decoder=PresetLatencyDecoder(1.0),
                      make_metrics=lambda e, cl, ch, f: [WindowLatencyBreakdown(cl)],
                      verbose=False)
    breakdown = r["metrics"]["window_latency"]
    rows = WindowLatencyBreakdown(r["cluster"]).rows()
    # every window decoded: 3 ops x ceil(11/3) = 12 windows
    assert breakdown["total"]["n"] == 12 and len(rows) == 12
    for row in rows:
        # the stages are non-negative and add up to the window's life
        assert row["buffer_fill"] >= 0 and row["dep_block"] >= 0
        assert row["queue_wait"] >= 0
        assert row["service"] == us(1.0)       # PresetLatencyDecoder's constant
        assert (row["buffer_fill"] + row["dep_block"] + row["queue_wait"]
                + row["service"]) == row["total"]
    # a chained (sequential) plan must show real dependency blocking somewhere
    assert breakdown["dep_block"]["max"] > 0


def test_breakdown_separates_queue_wait_from_dep_block():
    # one decoder unit + slow service: later windows wait IN QUEUE, not just on deps
    r = build_and_run(three_cnot_circuit(), num_units=1, d=3, rounds_per_op=11,
                      decoder=PresetLatencyDecoder(5.0),
                      make_metrics=lambda e, cl, ch, f: [WindowLatencyBreakdown(cl)],
                      verbose=False)
    assert r["metrics"]["window_latency"]["queue_wait"]["max"] > 0
