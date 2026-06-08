from __future__ import annotations
 
from typing import Any, TYPE_CHECKING
 
if TYPE_CHECKING:                      # hints only; lazy annotations mean no runtime import here
    from .message import Operation
    from .protocols import CodeModel
# ====================================================================
# LAYOUTS
# This module defines the qpu layout and code assignment so single 
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
        # identical to the original cluster computation: one code, all patches together.
        return self.code.spatial_nodes(len(op.qubits))
 
    def codes(self) -> list:
        return [self.code]
    
# TODO: This is for testing need to double check
class ZonedLayout:
    """A HETEROGENEOUS QPU: patches are assigned to codes by zone. Built from an explicit
    {patch_id: CodeModel} assignment plus a default code for any unlisted patch.
 
    Grounded in arXiv:2411.03202: a 'gross' bivariate-bicycle code as a dense memory zone and
    surface codes as compute zones, on one machine. The two rules below follow from each code
    being decoded separately (BB by BP+OSD, surface by MWPM):
 
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
