from __future__ import annotations
 
from typing import TYPE_CHECKING
 
from .message import Operation, Window, WindowPlan
 
if TYPE_CHECKING:                      
    from .protocols import DecodingScheme, LayoutModel

#==================================================================================================
# PLANNER
# The planner given the operation DAG and the code/layout/scheme, plans out the windows for each
# operation and tells the orchestrator about them. 
#==================================================================================================

#TODO: This is for testing only, remove it in the future
class FixedRounds:
    """Every operation runs the same number of rounds. FixedRounds(11) reproduces the original
    single global rounds_per_op (and keeps the regression trace byte-identical)."""
    def __init__(self, n: int):
        """Fix the per-operation round count."""
        self.n = int(n)
 
    def rounds_for(self, op, code) -> int:
        """Return the fixed round count (ignores op and code)."""
        return self.n

class CodeRounds:
    """Per-code rounds: each operation runs for its code's rounds_per_op() (the distance by
    default, a code may have more or fewer), optionally scaled."""
    def __init__(self, scale: float = 1.0):
        """Hold a multiplier on each code's intrinsic per-operation round count."""
        self.scale = scale
 
    def rounds_for(self, op, code) -> int:
        """Rounds = round(scale * code.rounds_per_op()), at least 1."""
        base = code.rounds_per_op() if hasattr(code, "rounds_per_op") \
            else code.rounds_per_logical_cycle()
        return max(1, int(round(self.scale * base)))

#TODO: This is for testing only, remove it in the future or update it
class WindowPlanner:
    """The default ExecutionPlanner the orchestrator's offline window/job planner.
 
    Given the operations, for each one it asks the DecodingScheme for the window layout (under
    that operation's own code, via the LayoutModel, so heterogeneous QPUs get per-zone windows),
    wires the inter-window dependency graph (intra-operation chain + each operation's first
    window depending on the last window of each predecessor), and records the decode-job size
    (spatial nodes) per operation. The result is a WindowPlan handed to the cluster ahead of
    time. This is pure compile-time work it runs before any syndrome flows and costs zero
    simulated ticks, same as the paper specifies (Impacts of Decoder Paper).
 
    The planning ALGORITHM lives here (not in the cluster), so swapping the planner e.g. a
    future parallel-window or spatial-window planner, changes the plan without touching the
    runtime cluster."""
    
    def __init__(self, scheme: DecodingScheme, layout: LayoutModel, rounds):
        """Hold the windowing policy (scheme), per-patch codes (layout), and the ROUNDS policy.
        `rounds` may be a RoundsPolicy or a plain int (wrapped as FixedRounds for back-compat)."""
        self.scheme = scheme
        self.layout = layout
        self.rounds_policy = rounds if hasattr(rounds, "rounds_for") else FixedRounds(int(rounds))
 
    def plan(self, ops: list[Operation]) -> WindowPlan:
        """Compute the full WindowPlan (windows, dependency graph, job sizes) for these ops."""
        opmap = {op.id: op for op in ops}
        # rounds-per-operation is per code via the ROUNDS policy (e.g. d for a distance-d code),
        # so each operation's windows are laid out under both its own code AND its own length.
        rounds = {oid: self.rounds_policy.rounds_for(op, self.layout.code_for_op(op))
                  for oid, op in opmap.items()}
        successors: dict = {oid: [] for oid in opmap}
        for oid, op in opmap.items():
            for p in op.predecessors:
                successors[p].append(oid)
        # the scheme decides each operation's window LAYOUT under ITS OWN code (per the layout)
        plans = {oid: self.scheme.plan_windows(oid, rounds[oid], self.layout.code_for_op(op))
                 for oid, op in opmap.items()}
        nwin = {oid: len(plans[oid]) for oid in opmap}
        windows: dict = {}
        op_windows: dict = {}
        for oid in opmap:
            for k, (commit_lo, commit_hi, buffer_hi) in enumerate(plans[oid]):
                n_rounds = (commit_hi - commit_lo + 1) + (buffer_hi - commit_hi)
                w = Window(op_id=oid, k=k, commit_lo=commit_lo, commit_hi=commit_hi,
                           buffer_hi=buffer_hi, n_rounds=n_rounds)
                windows[(oid, k)] = w
                op_windows.setdefault(oid, []).append(k)
        for oid, op in opmap.items():
            for k in range(nwin[oid]):
                w = windows[(oid, k)]
                if k > 0:
                    w.deps.append((oid, k - 1))                 # intra-op chain
                else:
                    for p in op.predecessors:
                        w.deps.append((p, nwin[p] - 1))         # predecessor's last window
                w.deps_remaining = len(w.deps)
        for key, w in windows.items():
            for dep in w.deps:
                windows[dep].dependents.append(key)
        spatial = {oid: self.layout.spatial_nodes_for(op) for oid, op in opmap.items()}
        rep = self.layout.codes()[0]                            # representative code (for summary)
        summary = dict(distance=rep.distance, commit=rep.commit_rounds(),
                       buffer=rep.buffer_rounds(),
                       rounds_per_op=(rounds[next(iter(opmap))] if opmap else 0),  # representative
                       windows_per_op=nwin.get(next(iter(opmap), 0), 0) if opmap else 0)
        return WindowPlan(windows=windows, nwin=nwin, op_windows=op_windows,
                          successors=successors, spatial_nodes=spatial,
                          total_windows=len(windows), summary=summary)
