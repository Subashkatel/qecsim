
from __future__ import annotations

from typing import Optional

from ..message import Operation


# ===========================================================================
# CIRCUIT FRONTEND
# this module defines the CircuitFrontend, which wraps a prebuilt 
# list of Operations as an InputFrontend. It also defines some example 
# circuits (three Clifford CNOTs, a mixed Clifford+T circuit, and a set of 
# independent T gates) built from Operation objects. The shared gate list 
# and classification logic is here too
# ===========================================================================
def _wire_circuit(ops: list[Operation]) -> list[Operation]:
    """Fill in each operation's patches / predecessors / has_successor from shared qubits.

    `ops` is in schedule order. Walking it once, for each qubit we remember the id of the most
    recent op that touched it; the next op on that qubit gains that previous one as a predecessor
    (and the previous one is marked as having a successor). Mutates in place and returns. Safe to
    call again on already-wired ops (idempotent)."""
    last_op_on_qubit = {}                      # qubit -> id of the most recent op that used it
    has_succ = {op.id: False for op in ops}
    preds = {op.id: set() for op in ops}
    for op in ops:                             # ops are in schedule order
        op.patches = tuple(op.qubits)          # one patch per qubit (default geometry)
        for q in op.qubits:
            if q in last_op_on_qubit:
                prev = last_op_on_qubit[q]
                preds[op.id].add(prev)
                has_succ[prev] = True
            last_op_on_qubit[q] = op.id
    for op in ops:
        op.predecessors = tuple(sorted(preds[op.id]))
        op.has_successor = has_succ[op.id]
    return ops


def three_cnot_circuit() -> list[Operation]:
    """The standard demo circuit -- three Clifford CNOTs:
        Op0: CNOT q0,q1     Op1: CNOT q2,q3     Op2: CNOT q1,q3
    Op0 and Op1 act on disjoint qubits, so they run in parallel; Op2 shares q1 (with Op0) and
    q3 (with Op1), so it depends on both and runs after them."""
    ops = [
        Operation(0, "Op0:CNOT(q0,q1)", (0, 1), clifford=True),
        Operation(1, "Op1:CNOT(q2,q3)", (2, 3), clifford=True),
        Operation(2, "Op2:CNOT(q1,q3)", (1, 3), clifford=True),
    ]
    return _wire_circuit(ops)


def cnot_plus_two_t_circuit() -> list[Operation]:
    """A mixed circuit that shows the non-Clifford stall:
        Op0: CNOT q0,q1   (Clifford)
        Op1: T q1         (non-Clifford, first in a chain)
        Op2: T q1         (non-Clifford, GATED by Op1's decoded outcome)."""
    ops = [
        Operation(0, "Op0:CNOT(q0,q1)", (0, 1), clifford=True),
        Operation(1, "Op1:T(q1)", (1,), clifford=False, gated_by=None),
        Operation(2, "Op2:T(q1)", (1,), clifford=False, gated_by=1),
    ]
    return _wire_circuit(ops)


def independent_t_circuit(n: int = 6) -> list[Operation]:
    """n independent (commuting) T gates, one per qubit, none gated on another. Their ONLY
    dependency is the supply of magic states -- this isolates the magic-state factory."""
    ops = [Operation(i, f"T(q{i})", (i,), clifford=False, gated_by=None)
           for i in range(n)]
    return _wire_circuit(ops)

def three_cnot_six_qubits_circuit() -> list[Operation]:
    """The standard demo circuit -- three Clifford CNOTs:
        Op0: CNOT q0,q1     Op1: CNOT q2,q3     Op2: CNOT q1,q3
    all of them are disjoint, so they run in parallel."""

    ops = [
        Operation(0, "Op0:CNOT(q0,q1)", (0, 1), clifford=True),
        Operation(1, "Op1:CNOT(q2,q3)", (3, 4), clifford=True),
        Operation(2, "Op2:CNOT(q1,q3)", (2, 5), clifford=True),
    ]
    return _wire_circuit(ops)


# ===========================================================================
# Input frontends (the InputFrontend seam). CircuitFrontend wraps a prebuilt op list;
# SurgeryIRFrontend reads a small line-based text IR. Both return list[Operation] via build(),
# so build_and_run(frontend=...) accepts either. To add a format, write a parser to the shared
# gate list (below). To add a gate, extend the sets below -- the single source of truth.
# OpenQASM input is deferred for now -- see the TODO at the bottom of this file.
# ===========================================================================
class CircuitFrontend:
    """The simplest InputFrontend: a thin wrapper around a Python-built operation list (e.g.
    three_cnot_circuit()), so the input is a swappable object rather than a hardcoded argument.
    SurgeryIRFrontend implements the same build() contract (OpenQASM support is a TODO)."""
    def __init__(self, ops: list[Operation]):
        """Hold a prebuilt operation list."""
        self.ops = ops

    def build(self) -> list[Operation]:
        """Return the operations (already wired; _wire_circuit is idempotent)."""
        return _wire_circuit(self.ops)


# The single source of truth for Clifford-ness. Extend a set to teach EVERY frontend a gate.
CLIFFORD_GATES = {"cnot", "cx", "h", "x", "y", "z", "s", "sdg", "cz", "swap", "id"}
NON_CLIFFORD_GATES = {"t", "tdg", "ccz", "ccx", "toffoli"}
ROTATION_GATES = {"rz", "rx", "ry", "p", "u1"}      # single-angle: Clifford depends on the angle
GENERAL_UNITARY_GATES = {"u2", "u3", "u"}           # multi-angle: treated as non-Clifford


def _parse_angle(expr) -> Optional[float]:
    """Parse a rotation angle into radians. Accepts a number directly (e.g. a qiskit gate
    parameter, already in radians) or a string in the pi-fraction forms TopQAD documents
    (n*pi, pi*n, n*pi/m, pi/n, f, f*pi, optional leading '-'). Returns None if it cannot be
    parsed, so the caller treats it as a worst-case non-Clifford rotation. No eval()."""
    import math
    if expr is None:
        return None
    if isinstance(expr, (int, float)):              # qiskit params arrive as floats (radians)
        return float(expr)
    s = str(expr).strip().lower().replace(" ", "")
    if not s:
        return None
    try:
        neg = s.startswith("-")
        if neg:
            s = s[1:]
        den = 1.0
        if "/" in s:                                # optional /denominator
            left, right = s.split("/", 1)
            den = math.pi if "pi" in right else float(right)
            s = left
        coeff = 1.0
        for part in s.split("*"):                   # numerator: product of factors
            coeff *= math.pi if part == "pi" else float(part)
        val = coeff / den
        return -val if neg else val
    except (ValueError, ZeroDivisionError):
        return None


def _rotation_is_clifford(angle_expr: Optional[str]) -> bool:
    """A single-axis rotation is Clifford iff its angle is a multiple of pi/2 (so rz(pi/2)=S is
    Clifford, rz(pi/4)=T is not). An unparseable angle is treated as non-Clifford (needs a T)."""
    import math
    a = _parse_angle(angle_expr)
    if a is None:
        return False
    k = a / (math.pi / 2.0)
    return abs(k - round(k)) < 1e-9


def _gate_is_clifford(mnemonic: str, angle: Optional[str] = None) -> bool:
    """Classify a gate. Raises on an unknown gate so unsupported input fails loudly rather than
    silently mis-modelling it -- the message names the sets to extend (the modularity hook)."""
    m = mnemonic.lower()
    if m in CLIFFORD_GATES:
        return True
    if m in NON_CLIFFORD_GATES or m in GENERAL_UNITARY_GATES:
        return False
    if m in ROTATION_GATES:
        return _rotation_is_clifford(angle)
    raise ValueError(f"unsupported gate '{mnemonic}' -- add it to CLIFFORD_GATES / "
                     f"NON_CLIFFORD_GATES / ROTATION_GATES / GENERAL_UNITARY_GATES")


def _ops_from_gatelist(gatelist: list) -> list[Operation]:
    """Shared lowering used by every text frontend: turn a parsed gate list into a wired
    operation DAG. Each entry is (mnemonic, qubit-tuple, is_clifford, gated_by). This is the
    one place Operation objects are built and `_wire_circuit` is applied, so each format stays
    a thin parser and they all behave identically downstream."""
    ops = []
    for i, (mnemonic, qubits, clifford, gated_by) in enumerate(gatelist):
        qstr = ",".join("q" + str(q) for q in qubits)
        ops.append(Operation(i, f"Op{i}:{mnemonic.upper()}({qstr})", tuple(qubits),
                             clifford=clifford, gated_by=gated_by))
    return _wire_circuit(ops)


class SurgeryIRFrontend:
    """InputFrontend for a small line-based LATTICE-SURGERY IR -- the paper's 'surgery IR' in
    miniature (lattice surgeries on named patches; arXiv:2511.10633 Sec III). One operation per
    line; '#' comments and blank lines are ignored. Grammar:

        CNOT qA qB          two-qubit Clifford      (also CZ, SWAP)
        H qA                single-qubit Clifford   (also X, Y, Z, S, SDG)
        T qA                non-Clifford (consumes a magic state)
        T qA gated_by N     non-Clifford gated by operation N's decoded outcome

    Implements build() -> list[Operation]. Unlike OpenQASM, this IR can express `gated_by`
    (the decode-feedback dependency)."""
    def __init__(self, text: str):
        """Hold the IR source text."""
        self.text = text

    def build(self) -> list[Operation]:
        """Parse the IR into the shared gate list, then lower it to the wired operation DAG."""
        gatelist = []
        for raw in self.text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            tok = line.split()
            mnemonic = tok[0]
            gated_by = None
            if "gated_by" in tok:
                gi = tok.index("gated_by")
                gated_by = int(tok[gi + 1])
                tok = tok[:gi]
            qubits = tuple(int(t[1:]) for t in tok[1:] if t.lower().startswith("q"))
            gatelist.append((mnemonic, qubits, _gate_is_clifford(mnemonic), gated_by))
        return _ops_from_gatelist(gatelist)


# ===========================================================================
# TODO: implement support for OpenQASM in the code.
# ===========================================================================
# An OpenQASM reader slots in here as another InputFrontend with the same
#   build() -> list[Operation]
# contract: parse OpenQASM 2.0 (TopQAD's input format, the qelib1.inc gate set) into the shared
# gate list and reuse _ops_from_gatelist + _gate_is_clifford above -- so it needs only a PARSER,
# no new lowering or gate-classification logic. The reference implementation (using the real
# qiskit qasm2 parser, plus a circuit=<QuantumCircuit> path for prebuilt/transpiled circuits)
# lives in section 11 of qec_des.py and can be lifted in unchanged when this is picked up.
# ===========================================================================
