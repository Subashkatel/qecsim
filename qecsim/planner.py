from __future__ import annotations
 
from typing import TYPE_CHECKING
 
from .message import Operation, Window, WindowPlan
 
if TYPE_CHECKING:                      
    from .protocols import DecodingScheme, LayoutModel

#==================================================================================================
# PLANNER
# The planner, given the operation DAG and the code/layout/scheme, plans out the windows for
# each operation and the dependency graph between them. This is the orchestrator's offline job
# from arXiv:2511.10633 Sec III: "the orchestrator determines the decoding windows for each
# surgery and the sequence of all decoding jobs required ... with their dependencies", and that
# plan is communicated to the decoder cluster AHEAD OF TIME, so planning costs zero simulated
# ticks and never sits on the reaction path.
#
# SEAM NOTE (windowing studies): the window LAYOUT and the INTRA-op dependency structure
# both come from the DecodingScheme -- via plan_windows plus the optional hooks wire_deps /
# entry_windows / exit_windows (see the DecodingScheme protocol). Without the hooks this
# planner falls back to the sequential chain, so plain schemes need not implement them;
# ParallelWindowScheme uses wire_deps for its A/B-layer structure (arXiv:2511.10633
# Sec II.4) with no planner change. Only the CROSS-op rule lives here, because only the
# planner sees the operation DAG. A planning algorithm that differs beyond what the hooks
# express (e.g. speculative early boundary exchange) swaps the whole planner via
# build_and_run(planner=...); the runtime cluster does not change either way.
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

class GateRounds:
    """Per-gate rounds under lattice surgery: a multi-qubit operation is modeled as the
    ZZ-then-XX merge pair (merge_steps * d rounds, merge_steps=2 by default); a
    single-qubit operation is one patch deformation (d rounds). Duration is set by fault
    tolerance (the merged stabilizers are repeated ~d times), NOT by qubit count -- a
    wider operation produces more data PER ROUND (the layout's spatial_nodes already
    scales with the op's qubits), not more rounds. Litinski (arXiv:1808.02892 Sec 1.1)
    performs ANY multi-patch measurement in one d-round time step: use merge_steps=1
    for that compilation."""
    def __init__(self, merge_steps: int = 2):
        """merge_steps: how many d-round surgery steps a multi-qubit op takes."""
        self.merge_steps = int(merge_steps)

    def rounds_for(self, op, code) -> int:
        """Multi-qubit ops take merge_steps * d rounds; single-qubit ops take d."""
        d = code.distance
        return self.merge_steps * d if len(op.qubits) >= 2 else d

#TODO: This is for testing only, remove it in the future or update it
class WindowPlanner:
    """The default ExecutionPlanner: the orchestrator's offline window/job planner.

    Given the operations, for each one it asks the DecodingScheme for the window layout
    (under that operation's own code, via the LayoutModel, so heterogeneous QPUs get
    per-zone windows) and for the intra-op dependency structure (the scheme's wire_deps /
    entry_windows / exit_windows hooks, falling back to the sequential chain), wires the
    cross-op edges (each operation's entry windows wait on its predecessors' exit
    windows), and records the decode-job size (spatial nodes) per operation. The result
    is a WindowPlan handed to the cluster ahead of time. This is pure compile-time work:
    it runs before any syndrome flows and costs zero simulated ticks, as arXiv:2511.10633
    Sec III specifies (the plan "is communicated to the decoder cluster ahead of time").

    The planning ALGORITHM lives here (not in the cluster), so swapping the planner --
    e.g. a future speculative-window or spatial-window planner -- changes the plan
    without touching the runtime cluster."""
    
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
            for k, spec in enumerate(plans[oid]):
                # a scheme returns 3-tuples (commit_lo, commit_hi, buffer_hi) or, for windows
                # with a LEADING buffer (parallel A/B scheme), 4-tuples prefixed by buffer_lo.
                if len(spec) == 3:
                    commit_lo, commit_hi, buffer_hi = spec
                    buffer_lo = commit_lo
                else:
                    buffer_lo, commit_lo, commit_hi, buffer_hi = spec
                n_rounds = buffer_hi - buffer_lo + 1
                w = Window(op_id=oid, k=k, commit_lo=commit_lo, commit_hi=commit_hi,
                           buffer_hi=buffer_hi, n_rounds=n_rounds, buffer_lo=buffer_lo)
                windows[(oid, k)] = w
                op_windows.setdefault(oid, []).append(k)
        # ---- dependency wiring. The INTRA-op structure belongs to the windowing scheme
        # (a sequential scheme chains its windows; the parallel A/B scheme has layer-B
        # windows depending on their two layer-A neighbours). The scheme provides it via
        # the optional hooks wire_deps / entry_windows / exit_windows; without them the
        # planner falls back to the original sequential chain, byte-identical to before.
        wire = getattr(self.scheme, "wire_deps", None)
        entry = getattr(self.scheme, "entry_windows", None)
        exits = getattr(self.scheme, "exit_windows", None)
        for oid, op in opmap.items():
            ws = [windows[(oid, k)] for k in range(nwin[oid])]
            if wire is not None:
                wire(ws)
            else:
                for k in range(1, nwin[oid]):
                    ws[k].deps.append((oid, k - 1))             # intra-op chain
            # CROSS-op wiring stays here (only the planner sees the DAG): each entry window
            # of this op waits for each exit window of every predecessor operation.
            for w_in in (entry(ws) if entry is not None else [ws[0]]):
                for p in op.predecessors:
                    pws = [windows[(p, k)] for k in range(nwin[p])]
                    for w_out in (exits(pws) if exits is not None else [pws[-1]]):
                        w_in.deps.append((p, w_out.k))
        for key, w in windows.items():
            w.deps_remaining = len(w.deps)
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
