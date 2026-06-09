from __future__ import annotations
 
from typing import TYPE_CHECKING
 
from .engine import Engine
from .message import Operation, SyndromePayload, Decision

if TYPE_CHECKING:                     
    from .cluster import DecoderCluster
    from .protocols import DeviceModel, Controller, MagicStateFactory

# =================================================================================================
# CHIP
# The QPU implementation this drives the syndrome round cadence emitting each round's payloads
# through the DeviceModel -> controller -> decoder cluster and advances the 
# operation DAG.
# Three main components:
#  - Dependency release 
#  - Magic state blocking
#  - Conditional operation blocking
# =================================================================================================

class Chip:
    """The default QPU drives the syndrome-round cadence, releases operations as 
    dependencies clear, gates non-Clifford gates until their correction returns, 
    and emits idle rounds while waiting."""
    def __init__(self, engine: Engine, device: DeviceModel, controller: Controller,
                 cluster: DecoderCluster, factory: MagicStateFactory,
                 round_ticks: int, rounds_per_op: int, code_distance: int):
        """Wire the chip to its device, controller, cluster, and factory; set the round cadence."""
        self.engine = engine
        self.device = device
        self.controller = controller
        self.cluster = cluster
        self.factory = factory
        self.round_ticks = round_ticks
        self.rounds_per_op = rounds_per_op    # how long an operation runs (temporal)
        self.code_distance = code_distance    # patch distance; also the buffer size
 
        self.ops: dict[int, Operation] = {}
        self.busy_qubits: set = set()
        self.requested: set[int] = set()           # ops whose start sequence has begun
        self.state_ready: set[int] = set()         # ops whose magic state is in hand (or none needed)
        self.started: set[int] = set()
        self.done_bodies: set[int] = set()
        self.gate_released: set[int] = set()       # for non-Clifford gating
        self.last_finish_time = 0
        # safety bound for the continuous idle-round emitter (a gate that never returns would
        # otherwise schedule forever); generous vs any realistic reaction time.
        self.MAX_IDLE_ROUNDS = 100 * code_distance
 
    def load(self, ops: list[Operation]) -> None:
        """Register all operations, then build the decoder windows."""
        for op in ops:
            self.ops[op.id] = op
            self.cluster.register_op(op)
        self.cluster.build_windows()          # cross-op deps need every op registered first
        self._try_start_all()
 
    # ---- starting operations ------------------------------------------------
    def _ready_to_attempt(self, op: Operation) -> bool:
        """True once this op's DATA dependencies are satisfied (its shared qubits are free).
        The REACTION dependency (a gated gate awaiting a prior decode) is handled separately in
        _maybe_begin, so the magic-state fetch can be issued concurrently with the reaction --
        the parallel supply-chain model of arXiv:2411.04270, where states sit pre-distilled in a
        buffer register rather than being fetched only after the reaction completes."""
        if op.id in self.started or op.id in self.requested:
            return False
        if any(q in self.busy_qubits for q in op.qubits):       # data dependency
            return False
        return True
 
    def _try_start_all(self) -> None:
        """Kick off the start sequence for any op whose data dependencies just cleared."""
        progress = True
        while progress:
            progress = False
            for op in list(self.ops.values()):
                if self._ready_to_attempt(op):
                    self._attempt_start(op)
                    progress = True
 
    def _attempt_start(self, op: Operation) -> None:
        # reserve the qubits now so nothing else grabs them while we wait
        """Reserve qubits and fetch the magic state (if any) -- IN PARALLEL with any pending
        reaction. The op physically begins only once BOTH the state is in hand and, if it is a
        gated gate, the prior decode has returned (see _maybe_begin)."""
        for q in op.qubits:
            self.busy_qubits.add(q)
        self.requested.add(op.id)
        if op.needs_magic_state:
            # Draw a distilled state from the factory supply chain. In the paper's model the
            # state is pre-distilled in a buffer register, so this fetch OVERLAPS the reaction
            # decode rather than stacking on top of it (arXiv:2411.04270).
            self.engine.log("Chip", f"{op.name} needs a magic state; asking the factory")
            self.factory.request(op.id, lambda o=op: self._on_state_ready(o))
        else:
            # Clifford ops AND factory-internal non-Clifford preparation need no distilled state.
            self._on_state_ready(op)
 
    def _on_state_ready(self, op: Operation) -> None:
        """The magic state (if any) is in hand; begin if the reaction dependency is met too."""
        self.state_ready.add(op.id)
        self._maybe_begin(op)
 
    def _maybe_begin(self, op: Operation) -> None:
        """Begin the op's physical rounds once BOTH hold: the magic state is in hand, and (for a
        blocked gate) the prior decode outcome has returned. These two waits run CONCURRENTLY, so a
        conditional T gate pays max(reaction, supply) -- not their sum, the old serial behavior."""
        if op.id in self.started or op.id not in self.state_ready:
            return
        if op.gated_by is not None and op.id not in self.gate_released:
            return                                              # reaction outcome not back yet
        self._begin(op)
 
    def _begin(self, op: Operation) -> None:
        """Run an operation's syndrome rounds, then mark its body done."""
        self.started.add(op.id)
        self.device.begin_operation(op)
        kind = "Clifford" if op.clifford else "NON-Clifford"
        gate = "" if op.gated_by is None else f" [released by op#{op.gated_by}]"
        self.engine.log("Chip", f"START {op.name}  ({kind}, qubits {op.qubits}){gate}")
        self.engine.schedule(self.round_ticks, lambda: self._round(op, 1),
                             label=f"round1({op.name})")
 
    # ---- per-round cadence --------------------------------------------------
    def _round(self, op: Operation, r: int) -> None:
        """Emit one syndrome round through the controller to the decoder cluster."""
        total = self.cluster.rounds_for(op)          # this op's length via the ROUNDS policy (D1)
        payload = self.device.round_payload(op, r)
        self.engine.log("Chip", f"{op.name} fires round {r}/{total}")
        # hand the round's syndromes to the controller, which relays them to the decoder
        self.controller.relay_syndrome(payload, self.cluster.on_syndrome_arrival)
        if r < total:
            self.engine.schedule(self.round_ticks, lambda: self._round(op, r + 1),
                                 label=f"round{r+1}({op.name})")
        else:
            self._body_done(op)
 
    def _body_done(self, op: Operation) -> None:
        """Op's physical work done; start successors and emit idle rounds while a gated successor waits."""
        self.done_bodies.add(op.id)
        self.last_finish_time = max(self.last_finish_time, self.engine.now)
        self.engine.log("Chip", f"{op.name} BODY DONE")
        for q in op.qubits:
            self.busy_qubits.discard(q)
        # try to start anything whose data dependency just cleared
        self._try_start_all()
        # If every operation's body is now physically complete, the QPU has finished all
        # its quantum work. (The decoder may still be draining its window queue.)
        if len(self.done_bodies) == len(self.ops):
            self.engine.log("Chip",
                            f"QPU FINISHED -- all {len(self.ops)} operations physically "
                            f"complete; chip now idle (decoder still draining)")
        # ---- IDLE-SYNDROME SEAM ----------------------------------------------------
        # A successor that is GATED on a decode result cannot run yet, so this op's patch idles
        # in storage, continuously measured. It emits one idle (memory) syndrome round PER round
        # for as long as the gated successor is still waiting -- NOT a fixed d-round burst. A
        # fixed burst would starve the decoder cluster if the reaction took longer than d rounds
        # (e.g. under contention the round trip can be several*d), misaligning the window buffers.
        # The recursive _emit_idle_round stops itself the moment the successor is released (its
        # gate returns in on_decision) or has started. Routed via on_memory_round (a count) as
        # before; switch to on_syndrome_arrival if you want the idle rounds actually decoded.
        # KNOWN SIMPLIFICATION: in arXiv:2511.10633 these memory stabilization rounds DO
        # require decoding (in parallel with the surgery decoding) and contribute decoder
        # load and storage error; here they only fill window buffers and are not decoded,
        # so decoder utilization during reaction waits is slightly UNDERSTATED.
        if op.has_successor and self._has_waiting_gated_successor(op.id):
            self.engine.log("Chip",
                            f"{op.name} patch idles (successor gated on a decode); "
                            f"emitting memory rounds every round until the correction returns")
            patch = op.patches[0] if op.patches else (op.qubits[0] if op.qubits else 0)
            self.engine.schedule(self.round_ticks,
                                 lambda oid=op.id, pt=patch: self._emit_idle_round(oid, pt, 1),
                                 label=f"idle-tick({op.name},1)")
 
    def _has_waiting_gated_successor(self, op_id: int) -> bool:
        """True while some successor on this op's patch is blocked and not yet released/started."""
        return any(
            (op_id in s.predecessors) and (s.gated_by is not None)
            and (s.id not in self.gate_released) and (s.id not in self.started)
            for s in self.ops.values()
        )
 
    def _emit_idle_round(self, op_id: int, patch, k: int) -> None:
        """Emit idle memory round k for an idling patch, then schedule the next one -- continuing
        every round until the gated successor is released (or has started), modelling a QPU that
        keeps measuring stabilizers while it waits in storage. Self-terminating + capped."""
        if k > self.MAX_IDLE_ROUNDS or not self._has_waiting_gated_successor(op_id):
            return                                          # gate returned (or started): stop
        self.controller.relay_syndrome(
            SyndromePayload(op_id, patch, k),
            lambda p, o=op_id: self.cluster.on_memory_round(o))
        self.engine.schedule(self.round_ticks,
                             lambda oid=op_id, pt=patch, kk=k: self._emit_idle_round(oid, pt, kk + 1),
                             label=f"idle-tick({self.ops[op_id].name},{k + 1})")
 
    # ---- decode result came back (only relevant for blocked T gates) ----------
    def on_decision(self, decision: Decision) -> None:
        """A correction came back: release the blocked gate. It begins immediately if its magic
        state has already arrived (fetched in parallel during the reaction); otherwise it begins
        when the state lands. _maybe_begin enforces the AND of the two conditions."""
        self.gate_released.add(decision.gadget_id)
        target = self.ops[decision.gadget_id]
        self.engine.log("Chip",
                        f"received basis '{decision.basis}' -> UNBLOCKS {target.name}; trying to start")
        self._maybe_begin(target)      # released op may begin now (state may already be buffered)
        self._try_start_all()          # and anything whose data deps just cleared
 