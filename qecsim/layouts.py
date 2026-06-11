from __future__ import annotations
 
from typing import Any, TYPE_CHECKING
 
if TYPE_CHECKING:                      # hints only; lazy annotations mean no runtime import here
    from .message import Operation
    from .protocols import CodeModel
# ====================================================================
# LAYOUTS
# This module defines the QPU layout and code assignment, so a single
# machine can be heterogeneous.
# ====================================================================
# TODO: This is for testing need to double check
class UniformLayout:
    """The whole QPU uses ONE code -- one zone, every patch the same. This reproduces the
    original single-code behavior exactly: spatial_nodes_for(op) is the code's aggregate
    spatial_nodes(num_patches), and code_for_op is always that one code."""
    def __init__(self, code: CodeModel):
        self.code = code
 
    @property
    def name(self) -> str:
        return f"uniform[{self.code.name}]"
 
    @property
    def distance(self) -> int:
        return self.code.distance
 
    def code_for_patch(self, patch_id: Any) -> CodeModel:
        return self.code
 
    def code_for_op(self, op: Operation) -> CodeModel:
        return self.code
 
    def spatial_nodes_for(self, op: Operation) -> int:
        # decode-graph size scales with the op's patches (qubits if never wired)
        num_patches = len(op.patches) if op.patches else len(op.qubits)
        return self.code.spatial_nodes(num_patches)
 
    def codes(self) -> list:
        return [self.code]
    
# TODO: This is for testing need to double check
class ZonedLayout:
    """A HETEROGENEOUS QPU: patches are assigned to codes by zone. Built from an explicit
    {patch_id: CodeModel} assignment plus a default code for any unlisted patch.

    Grounded in arXiv:2411.03202 (HetEC): surface-code blocks handle compute ("Clifford and
    non-Clifford computations assisted by magic states") while gross-code [[144,12,12]]
    blocks serve as dense logical memory, joined by an ancilla bus (103 physical qubits in
    the paper, Sec 2.4.1) for inter-code data movement via automorphism + teleportation
    (Sec 3.2); headline saving up to 6.42x fewer physical qubits for up to 3.43x more
    execution time. (arXiv:2604.06319 pushes the same heterogeneous idea to a 138x qubit
    reduction under detailed accounting.) That paper leaves decoder coordination between
    codes unspecified, so the two rules below are THIS simulator's modeling choices, which
    follow from each code being decoded separately (BB by BP, surface by MWPM) -- the G1
    per-code decoder map in the cluster:
 
      - code_for_op(op): the MOST DEMANDING (largest-distance) code among the operation's
        patches. A single-zone op -> that zone's code. A zone-straddling op (a movement/merge
        between, say, surface and BB) -> the larger code, because the decode window must cover
        the slowest patch.
      - spatial_nodes_for(op): GROUP the op's patches by code, and for each code sum that
        code's aggregate spatial_nodes(group_size). Separate codes are separate decoding
        graphs, so their node counts add. (With one group this reduces to UniformLayout.)
    """
    def __init__(self, assignment: dict, default: CodeModel):
        self.assignment = dict(assignment)
        self.default = default
 
    @property
    def name(self) -> str:
        names = sorted({c.name for c in self.codes()})
        return "zoned[" + " + ".join(names) + "]"
 
    @property
    def distance(self) -> int:
        # representative distance for the summary log: the default zone's distance.
        return self.default.distance
 
    def code_for_patch(self, patch_id: Any) -> CodeModel:
        return self.assignment.get(patch_id, self.default)
 
    def _op_patches(self, op: Operation):
        return op.patches if op.patches else op.qubits
 
    def code_for_op(self, op: Operation) -> CodeModel:
        codes = [self.code_for_patch(p) for p in self._op_patches(op)]
        if not codes:
            return self.default
        return max(codes, key=lambda c: c.distance)          # window covers the slowest patch
 
    def spatial_nodes_for(self, op: Operation) -> int:
        groups: dict = {}                                    # code -> number of patches in it
        for p in self._op_patches(op):
            c = self.code_for_patch(p)
            groups[c] = groups.get(c, 0) + 1
        return sum(c.spatial_nodes(n) for c, n in groups.items())
 
    def codes(self) -> list:
        seen, out = set(), []
        for c in list(self.assignment.values()) + [self.default]:
            if id(c) not in seen:
                seen.add(id(c))
                out.append(c)
        return out
