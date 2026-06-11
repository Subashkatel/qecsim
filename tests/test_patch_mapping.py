"""The qubit -> patch mapping (how many logical qubits live in one patch).

A patch is the hardware region the decoder watches as one picture. The surface
code puts ONE logical qubit in each patch; block codes like the gross code put
12 in one patch, and a patch can only do one operation at a time. The frontends
therefore accept `qubit_to_patch`, a dict saying which patch each logical qubit
lives in. Leaving it out means every qubit is its own patch -- byte-identical to
the old behavior."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from conftest import trace_time
from qecsim.decoders import PresetLatencyDecoder
from qecsim.frontends.circuit import CircuitFrontend, _wire_circuit
from qecsim.layouts import UniformLayout
from qecsim.codes import SurfaceCodeModel
from qecsim.message import Operation
from qecsim.wiring import build_and_run


TWO_QUBITS_PER_PATCH = {0: "patch0", 1: "patch0", 2: "patch1", 3: "patch1"}


def _run(ops):
    r = build_and_run(ops, num_units=2, d=3, rounds_per_op=11,
                      decoder=PresetLatencyDecoder(1.0), verbose=False)
    return r["engine"].log_lines


def test_default_every_qubit_is_its_own_patch():
    ops = CircuitFrontend([
        Operation(0, "A:CNOT(q0,q1)", (0, 1), clifford=True),
    ]).build()
    assert ops[0].patches == (0, 1)


def test_two_qubits_in_one_patch_share_it():
    """An op touching two qubits of the SAME patch involves one patch, not two."""
    ops = CircuitFrontend([
        Operation(0, "A:CNOT(q0,q1)", (0, 1), clifford=True),
    ], qubit_to_patch=TWO_QUBITS_PER_PATCH).build()
    assert ops[0].patches == ("patch0",)
    # and the decode job is sized for one patch, not two
    layout = UniformLayout(SurfaceCodeModel(d=3))
    assert layout.spatial_nodes_for(ops[0]) == SurfaceCodeModel(d=3).spatial_nodes(1)


def test_ops_on_same_patch_take_turns():
    """q0 and q1 live in the same patch, so ops on them must run in program order
    even though they touch different qubits (a patch does one operation at a time)."""
    ops = CircuitFrontend([
        Operation(0, "A:M(q0)", (0,), clifford=True),
        Operation(1, "B:M(q1)", (1,), clifford=True),
    ], qubit_to_patch=TWO_QUBITS_PER_PATCH).build()
    assert ops[1].predecessors == (0,)
    lines = _run(ops)
    assert trace_time(lines, "START B:M(q1)") >= trace_time(lines, "A:M(q0) BODY DONE")


def test_ops_on_different_patches_run_in_parallel():
    """q0 and q2 live in different patches: no dependency, simultaneous starts."""
    ops = CircuitFrontend([
        Operation(0, "A:M(q0)", (0,), clifford=True),
        Operation(1, "B:M(q2)", (2,), clifford=True),
    ], qubit_to_patch=TWO_QUBITS_PER_PATCH).build()
    assert ops[1].predecessors == ()
    lines = _run(ops)
    assert trace_time(lines, "START A:M(q0)") == trace_time(lines, "START B:M(q2)")


def test_rewiring_without_mapping_keeps_block_patches():
    """Regression: re-running the wiring with no mapping must NOT erase a block
    assignment (it used to silently reset every qubit to its own patch)."""
    ops = CircuitFrontend([
        Operation(0, "A:M(q0)", (0,), clifford=True),
        Operation(1, "B:M(q1)", (1,), clifford=True),
    ], qubit_to_patch=TWO_QUBITS_PER_PATCH).build()
    _wire_circuit(ops)                         # no mapping -- must keep the blocks
    assert ops[0].patches == ("patch0",)
    assert ops[1].predecessors == (0,)


def test_op_listing_same_qubit_twice_is_rejected():
    """A malformed op like CNOT(q0,q0) must fail with a message naming the real
    problem, not a confusing wiring complaint."""
    with pytest.raises(ValueError, match="more than once"):
        CircuitFrontend([Operation(0, "X(q0,q0)", (0, 0), clifford=True)]).build()


def test_mapping_missing_a_qubit_is_rejected():
    """A qubit the mapping forgot must be named in the error, not a bare KeyError."""
    with pytest.raises(ValueError, match="no patch for qubit"):
        CircuitFrontend([Operation(0, "A:M(q9)", (9,), clifford=True)],
                        qubit_to_patch=TWO_QUBITS_PER_PATCH).build()
