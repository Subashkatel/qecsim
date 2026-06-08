from qecsim.protocols import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from message import Window, Operation
    from protocols import CodeModel, LayoutModel

# ===============================================================================
# SCHEMES
# This is the default decodign scheme for now it answers two questions:
# 1. how an operation syndrome rounds are grouped into commit and buffer windows
# 2. when a window has accumulated enough syndrome rounds to be decoded safely.
# It is a pure POLICY -- no engine, no clock -- so it is clean swap point for 
# other windowing schemes like adaptive windowing, speculative window decoding
# and so on.
# ===============================================================================

class SlidingWindowScheme:
    """Default sliding window scheme is a simple fixed-size window. Each operation is 
    chopped into windows that commit c rounds behind a b round look-ahead buffer where
    c and b come form the codes (currently both are the code distance d). if the buffer
    spills goes beyond the end of the operations own rounds, the overflow is gotten form 
    the sucessors operations early rounds(or idel memory rounds).Once a window commits, the
    error strings crossing into its buffer become "artificial defects" handed to the next
    window as boundary bits """

    def plan_windows(self, op_id: int, n_rounds: int, code: CodeModel) -> list[tuple[int, int]]:
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
