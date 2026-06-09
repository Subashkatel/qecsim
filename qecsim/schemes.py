from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .message import Window, Operation
    from .protocols import CodeModel, LayoutModel

# ===============================================================================
# SCHEMES
# This is the default decoding scheme. It answers two questions:
# 1. how an operation's syndrome rounds are grouped into commit and buffer windows
# 2. when a window has accumulated enough syndrome rounds to be decoded safely.
# It is a pure POLICY -- no engine, no clock -- so it is a clean swap point for
# other windowing schemes: adaptive windowing (ADaPT, arXiv:2605.01149), parallel
# A/B-layer windowing (arXiv:2511.10633 Sec II.4), speculative windowing, double-
# window decoder switching (arXiv:2510.25222 Sec III.3), and so on.
# ===============================================================================

class SlidingWindowScheme:
    """The standard SEQUENTIAL (forward) sliding-window decoder: each operation is chopped
    into windows that commit C rounds behind a B-round look-ahead buffer, where C and B come
    from the code (both default to the code distance d). If the buffer spills beyond the end
    of the operation's own rounds, the overflow comes from the successor operation's early
    rounds (or idle memory rounds). Once a window commits, error strings crossing into its
    buffer become "artificial defects" handed to the NEXT window as its boundary -- so the
    windows of one operation form a serial dependency chain.

    Grounding. This is the (W, C)-sliding window of the literature: window size W = C + B
    with the commit-then-carry-forward artificial-defect rule (ADaPT, arXiv:2605.01149
    Sec II-C; used on FPGA hardware for the gross code with W = d and runtime-configurable C,
    arXiv:2510.21600 Sec 4.1). It is NOT the parallel windowing of arXiv:2511.10633 Sec II.4,
    where 3d-round windows have buffer/commit/buffer sub-regions and alternate in two layers
    (independent layer-A windows decode concurrently, then layer-B windows consume their
    boundaries; memory reaction time gamma_mem = 6d*tau_d(d^2) + t_com, Eq. 13). The serial
    chain here is throughput-limited by one window's decode per C rounds; the parallel scheme
    exists precisely to break that chain. Implementing it means a new DecodingScheme PLUS
    dependency wiring in the planner (see WindowPlanner's note on where deps are wired)."""

    def plan_windows(self, op_id: int, n_rounds: int, code: CodeModel) -> list[tuple[int, int, int]]:
        """Lay out the windows for an operations: commit d rounds behind a d round look-ahead buffer."""
        import math
        C,B,R = code.commit_rounds(), code.buffer_rounds(), n_rounds
        nwin = max(1, math.ceil(R / C))
        plan = []
        for k in range(nwin):
            commit_lo = k * C + 1
            commit_hi = min((k + 1) * C, R)
            buffer_hi = commit_hi + B
            plan.append((commit_lo, commit_hi, buffer_hi))
        return plan
    
    def data_complete(self, window: "Window", rounds_arrived: int, successor_rounds: int,
                      memory_rounds: int, n_rounds: int, has_successor: bool,
                      op: "Operation" = None, layout: "LayoutModel" = None) -> bool:
        """Return True once the commit+buffer rounds have arrived (including spillover from successor or memory rounds if needed).
        This is the default and its just temporal rule op/layout is avaiable for more complex schemes currently ignored. """
        in_op_need = min(window.buffer_hi, n_rounds)       # commit + in-op buffer rounds
        if rounds_arrived < in_op_need:
            return False
        overflow = window.buffer_hi - n_rounds  # buffer rounds beyond the end of the operation
        if overflow > 0:
            if not has_successor:
                return True # no successor to provide overflow rounds, so just go with what we have
            return (successor_rounds >= overflow) or (memory_rounds >= overflow) # successor or memory rounds can provide the overflow
        return True
