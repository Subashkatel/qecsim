from __future__ import annotations
 
from collections import deque
from typing import Callable, Optional, TYPE_CHECKING
 
from engine import Engine
from message import Operation, DecodeResult, Decision, WindowPlan
from qecsim import engine
 
if TYPE_CHECKING:                      # type-only; the controller is wired in at runtime via connect()
    from protocols import Controller

#TODO: Write what is does
# =====================================================================================
# ORCHESTRATORS
# Paper does multiple things 
# =====================================================================================

class PauliFrameOrchestrator:
    """The default orchestrator implementation, which just tracks the Pauli frame and feeds it back to the controller as needed."""

    def __init__(self, engine: Engine, history_size: int = 512, retain_all: bool = False):
        """Hold the Pauli frame and the map of which gate each operation gates."""

        self.engine = engine
        self.pauli_frame: dict = {}
        self.outcomes: dict[int, int] = {}
        # which op (if any) is gated by op_id:
        self.gated_by_index: dict[int, int] = {}
        # --- retained-but-bounded debugging record (see class/ctor docstring) ---
        self.history: deque = deque(maxlen=history_size)   # recent results, fixed memory
        self.stats: dict[str, int] = {"frame_updates": 0, "outcomes": 0, "decisions": 0}
        self.retain_all = retain_all
        self.archive: Optional[dict] = {} if retain_all else None
        # Job-4 dispatch plumbing, filled by connect() at wiring time. Left None until then so
        # the frame core stays usable standalone (e.g. tests / latency-only runs that call
        # on_result directly and never dispatch).
        self.controller: Optional["Controller"] = None
        self.decision_sink: Optional[Callable] = None

    def connect(self, controller: "Controller", decision_sink: Callable) -> None:
        """Connect the orchestrator to the controller and tells where the controller 
        should send the decisions it returns. """
        self.controller = controller
        self.decision_sink = decision_sink

    def register_gate(self, gated_op_id: int, gating_op_id: int) -> None:
        """Record that operation `blocked` waits on the decode result of `blocked`. ONE blocking op
        may block MULTIPLE successors -- a single measurement outcome can feed-forward to several
        conditional gates -- so successors accumulate in a list."""
        self.gated_by_index.setdefault(gating_op_id, []).append(gated_op_id)
 
    def announce_plan(self, plan: "WindowPlan") -> None:
        """handoff: the orchestrator has compiled the execution plan and now sends it to
        the decoder cluster. This happens at COMPILE TIME -- 0 simulated ticks, off the reaction
        path (arXiv:2511.10633 Sec III), so it never contributes to reaction time."""
        self.engine.log("Orchestrator",
                        f"compiled execution plan off the reaction path (0 ticks); sending "
                        f"{plan.total_windows} windows across {len(plan.nwin)} operation(s) "
                        f"to the decoder cluster ahead of time")
 
    def _record_and_gc(self, op: Operation, kind: str, outcome: int,
                       basis: Optional[str] = None) -> None:
        """Save a compact record of this result (for debugging / audit), then free the live
        per-op frame/outcome state. The live dicts otherwise grow with the circuit an
        unbounded leak at utility scale."""
        
        # This records the result in a compact form for debugging and audit.
        rec = {"t": self.engine.now, "op_id": op.id, "name": op.name,
               "kind": kind, "outcome": outcome, "basis": basis}
        self.history.append(rec)
        if self.archive is not None:
            self.archive[op.id] = rec
        self.stats["frame_updates" if kind == "frame_update"
                   else "decisions" if kind == "decision" else "outcomes"] += 1
        self.outcomes.pop(op.id, None)
        self.pauli_frame.pop(op.id, None)
 
    def integrate(self, op: Operation, result: DecodeResult) -> None:
        """Receive a decoded result and save it into the Pauli frame via on_result, then
        Send every conditional decision it unblocks back to the chip THROUGH the controller
        (orchestrator -> controller -> chip). on_result stays the pure, overridable frame core;
        integrate is the owned dispatch wrapped around it. With no controller/sink connected (a
        frame-only or latency-only run) it updates the frame and dispatches nothing."""
        for decision in self.on_result(op, result):
            if self.controller is None or self.decision_sink is None:
                continue
            # A conditional instruction must physically reach the chip. The orchestrator owns
            # this: it relays the decision over the controller (which logs itself as a hop) to
            # the chip, which has been blocking on it and now unblocks. A gating op may emit
            # several decisions (fan-out); each is dispatched in turn.
            self.engine.log("Orchestrator",
                            f"DISPATCH conditional for op#{decision.gadget_id}: "
                            f"basis '{decision.basis}' -> controller -> chip")
            self.controller.relay_instruction(decision, self.decision_sink)
 
    def on_result(self, op: Operation, result: DecodeResult) -> list[Decision]:
        """Save an outcome into the Pauli frame; return a Decision for EACH gated successor. A
        single outcome can feed-forward to several conditional gates, so this returns a
        LIST -- empty for a Clifford result or a non-Clifford op that gates nothing."""
        
        outcome = result.logical_value if result.logical_value is not None else 1
        self.outcomes[op.id] = outcome
        if op.clifford:
            # Just a frame update. It stays in the orchestrator; NOTHING is sent back
            # to the QPU (a Pauli-frame correction is classical and applied lazily).
            # Only the decoder->orchestrator hop (t_do) was paid; t_oc + t_cq are not.
            self.pauli_frame[op.id] = "frame-updated"
            self.engine.log("Orchestrator",
                            f"result for {op.name}: Pauli-frame update (Clifford) -- "
                            f"stays here, no instruction returns to the QPU")
            self._record_and_gc(op, "frame_update", outcome)
            return []
        # Non-Clifford: later gadget(s) need this outcome to pick their basis.
        self.engine.log("Orchestrator",
                        f"result for {op.name}: non-Clifford outcome decoded")
        gated_ops = self.gated_by_index.pop(op.id, [])     # ALL successors this op gates
        if gated_ops:
            basis = "X" if outcome else "Z"                # same outcome -> same basis for all
            tgt = ", ".join(f"op#{g}" for g in gated_ops)
            self.engine.log("Orchestrator",
                            f"  -> decides basis '{basis}' for {tgt} and UNBLOCKS the chip")
            self._record_and_gc(op, "decision", outcome, basis)
            return [Decision(gadget_id=g, basis=basis) for g in gated_ops]
        self._record_and_gc(op, "outcome", outcome)        # no gated successor reads this outcome
        return []