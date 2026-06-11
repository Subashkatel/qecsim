#==================================================================
# TESTS FOR INSTRUMENTATION (window-lifecycle latency breakdown)
#==================================================================
from qecsim.config import us
from qecsim.decoders import PresetLatencyDecoder
from qecsim.frontends.circuit import cnot_plus_two_t_circuit, three_cnot_circuit
from qecsim.metrics import BacklogTrajectory, WindowLatencyBreakdown
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


# ---- per-gate backlog (the r_i of arXiv:2510.25222) -------------------------------------

def test_backlog_trajectory_measures_the_gated_t_gate():
    """cnot_plus_two_t has exactly ONE gated gate (the second T waits on the first T's
    decode): one row, a positive reaction wait, and a backlog of wait-in-rounds plus
    the gate's own rounds."""
    r = build_and_run(cnot_plus_two_t_circuit(), num_units=2, d=3, rounds_per_op=11,
                      decoder=PresetLatencyDecoder(1.0),
                      make_metrics=lambda e, cl, ch, f: [BacklogTrajectory(ch)],
                      verbose=False)
    res = r["metrics"]["backlog_trajectory"]
    rows = BacklogTrajectory(r["chip"]).rows()
    assert res["n"] == 1 and len(rows) == 1
    assert rows[0]["wait"] > 0                              # reaction is never free
    assert rows[0]["backlog_rounds"] == rows[0]["wait"] / r["chip"].round_ticks + 11


def test_backlog_trajectory_registration_changes_nothing():
    """The metric only reads chip timestamps: a run with it registered produces the
    byte-identical trace of a run without it."""
    bare = build_and_run(cnot_plus_two_t_circuit(), num_units=2, d=3, rounds_per_op=11,
                         decoder=PresetLatencyDecoder(1.0), verbose=False)
    metered = build_and_run(cnot_plus_two_t_circuit(), num_units=2, d=3, rounds_per_op=11,
                            decoder=PresetLatencyDecoder(1.0),
                            make_metrics=lambda e, cl, ch, f: [BacklogTrajectory(ch)],
                            verbose=False)
    assert bare["engine"].log_lines == metered["engine"].log_lines


def test_backlog_grows_when_the_decoder_is_too_slow():
    """A slower decoder must show a LARGER reaction wait for the same gated gate --
    the signal every backlog/divergence study reads off this metric."""
    waits = {}
    for lat in (1.0, 50.0):
        r = build_and_run(cnot_plus_two_t_circuit(), num_units=2, d=3, rounds_per_op=11,
                          decoder=PresetLatencyDecoder(lat),
                          make_metrics=lambda e, cl, ch, f: [BacklogTrajectory(ch)],
                          verbose=False)
        waits[lat] = r["metrics"]["backlog_trajectory"]["max_wait"]
    assert waits[50.0] > waits[1.0]


def test_backlog_rows_cover_fan_out_gating():
    """ONE decoded outcome can release SEVERAL gated gates: one row each, and they
    share the same reaction wait (same gating decode, same dispatch event)."""
    from qecsim.frontends.circuit import CircuitFrontend
    from qecsim.message import Operation
    ops = CircuitFrontend([
        Operation(0, "A:T(q0)", (0,), clifford=False),
        Operation(1, "B:T(q1)", (1,), clifford=False, gated_by=0),
        Operation(2, "C:T(q2)", (2,), clifford=False, gated_by=0),
    ]).build()
    r = build_and_run(ops, num_units=2, d=3, rounds_per_op=11,
                      decoder=PresetLatencyDecoder(1.0),
                      make_metrics=lambda e, cl, ch, f: [BacklogTrajectory(ch)],
                      verbose=False)
    rows = BacklogTrajectory(r["chip"]).rows()
    assert len(rows) == 2
    assert rows[0]["wait"] == rows[1]["wait"] > 0


def test_backlog_rounds_use_the_ops_own_cadence():
    """A per-code round_us override changes how many rounds fit in a wait; the metric
    must divide by the OP'S cadence, not the chip's global one."""
    from qecsim.codes import SurfaceCodeModel
    from qecsim.frontends.circuit import CircuitFrontend
    from qecsim.message import Operation
    fast = SurfaceCodeModel(d=3, round_us=0.5)             # != the global 1.1 us
    ops = CircuitFrontend([
        Operation(0, "A:T(q0)", (0,), clifford=False),
        Operation(1, "B:T(q0)", (0,), clifford=False, gated_by=0),
    ]).build()
    r = build_and_run(ops, num_units=2, d=3, rounds_per_op=11, code=fast,
                      decoder=PresetLatencyDecoder(1.0),
                      make_metrics=lambda e, cl, ch, f: [BacklogTrajectory(ch)],
                      verbose=False)
    row = BacklogTrajectory(r["chip"]).rows()[0]
    assert row["backlog_rounds"] == row["wait"] / us(0.5) + 11
