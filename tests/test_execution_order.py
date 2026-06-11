"""Program-order execution (the chip's dependency-DAG release).

Regression for a real bug: the chip used to start an operation whenever its qubits
were momentarily free (greedy busy_qubits reservation), which reordered non-commuting
gates -- a T(q0) ran before an earlier CNOT(q0,q1) that was still waiting on q1,
executing a physically different circuit. The fix: an op starts only when every
operation it depends on (per-qubit program order, op.predecessors -- the trivial rule
of arXiv:2405.17688, the dependency rule of dascot arXiv:2311.18042) has finished its
body. Qubit reservation remains as a fail-loud invariant, not a scheduler."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from conftest import trace_time
from qecsim.decoders import PresetLatencyDecoder
from qecsim.frontends.circuit import CircuitFrontend
from qecsim.message import Operation
from qecsim.planner import GateRounds
from qecsim.wiring import build_and_run


def _run(ops, **kw):
    kw.setdefault("num_units", 2)
    kw.setdefault("d", 3)
    kw.setdefault("decoder", PresetLatencyDecoder(1.0))
    r = build_and_run(ops, verbose=False, **kw)
    return r["engine"].log_lines


def test_t_gate_waits_for_earlier_cnot_on_same_qubit():
    """The minimal reproduction of the bug. Program order on q0 is A-then-C via B's
    qubit chain: A=CNOT(1,2) holds q1; B=CNOT(0,1) must wait for A; C=T(0) comes
    after B on q0. The greedy chip started C at t=0 (q0 momentarily free) --
    reordering non-commuting gates. C must start only after B's body is done."""
    ops = CircuitFrontend([
        Operation(0, "A:CNOT(q1,q2)", (1, 2), clifford=True),
        Operation(1, "B:CNOT(q0,q1)", (0, 1), clifford=True),
        Operation(2, "C:T(q0)", (0,), clifford=False),
    ]).build()
    lines = _run(ops, rounds_per_op=11)
    b_done = trace_time(lines, "B:CNOT(q0,q1) BODY DONE")
    c_start = trace_time(lines, "START C:T(q0)")
    assert c_start >= b_done


def test_order_holds_under_heterogeneous_durations():
    """Stage-2 evidence: dependency release is duration-blind. Same circuit under
    GateRounds (CNOTs 2d rounds, T d rounds) -- the order must still hold even
    though every op now has a different length."""
    ops = CircuitFrontend([
        Operation(0, "A:CNOT(q1,q2)", (1, 2), clifford=True),
        Operation(1, "B:CNOT(q0,q1)", (0, 1), clifford=True),
        Operation(2, "C:T(q0)", (0,), clifford=False),
    ]).build()
    lines = _run(ops, rounds_policy=GateRounds())
    assert trace_time(lines, "START C:T(q0)") >= trace_time(lines, "B:CNOT(q0,q1) BODY DONE")


def test_brickwork_with_t_keeps_program_order():
    """The shape that exposed the bug at scale: brickwork layers + a T on q0. The
    universal invariant: EVERY op starts at-or-after EVERY one of its predecessors'
    bodies finished. (On the greedy chip, L2:CNOT(q0,q1) jumped ahead of
    L1:CNOT(q1,q2) -- same-qubit ops reordered -- which this catches.)"""
    ops, oid = [], 0
    for layer in range(4):
        for q in range(layer % 2, 7, 2):
            ops.append(Operation(oid, f"L{layer}:CNOT(q{q},q{q+1})", (q, q + 1),
                                 clifford=True))
            oid += 1
    ops.append(Operation(oid, "T(q0)", (0,), clifford=False))
    ops = CircuitFrontend(ops).build()
    lines = _run(ops, rounds_per_op=11, num_units=8)
    opmap = {op.id: op for op in ops}
    for op in ops:
        for p in op.predecessors:
            assert trace_time(lines, f"START {op.name}") >= \
                   trace_time(lines, f"{opmap[p].name} BODY DONE"), \
                f"{op.name} started before its predecessor {opmap[p].name} finished"


def test_unwired_conflicting_ops_fail_loudly():
    """The busy_qubits invariant: ops sharing a qubit WITHOUT a dependency edge
    (someone skipped _wire_circuit) must raise, not silently serialize or overlap."""
    ops = [Operation(0, "A:X(q0)", (0,), clifford=True),
           Operation(1, "B:X(q0)", (0,), clifford=True)]   # deliberately NOT wired
    with pytest.raises(RuntimeError, match="share"):
        build_and_run(ops, num_units=1, d=3, rounds_per_op=11,
                      decoder=PresetLatencyDecoder(1.0), verbose=False)


def test_raw_op_listing_same_qubit_twice_names_the_real_problem():
    """An unwired op like X(q0,q0) reaching the chip directly must be diagnosed as a
    malformed op, not blamed on missing wiring."""
    ops = [Operation(0, "X(q0,q0)", (0, 0), clifford=True)]   # bypasses the frontends
    with pytest.raises(RuntimeError, match="more than once"):
        build_and_run(ops, num_units=1, d=3, rounds_per_op=11,
                      decoder=PresetLatencyDecoder(1.0), verbose=False)


def test_parallel_ops_still_run_in_parallel():
    """The fix must not serialize what is genuinely independent: two disjoint CNOTs
    still start at the same time."""
    ops = CircuitFrontend([
        Operation(0, "A:CNOT(q0,q1)", (0, 1), clifford=True),
        Operation(1, "B:CNOT(q2,q3)", (2, 3), clifford=True),
    ]).build()
    lines = _run(ops, rounds_per_op=11)
    assert trace_time(lines, "START A:CNOT(q0,q1)") == trace_time(lines, "START B:CNOT(q2,q3)")
