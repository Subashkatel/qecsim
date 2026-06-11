from __future__ import annotations
 
from typing import Optional, TYPE_CHECKING
 
from .config import us
from .engine import Engine
from .message import Operation, SyndromePayload, Decision

if TYPE_CHECKING:
    from .protocols import (DeviceModel, Controller, MagicStateFactory,
                            WorkloadManager)

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
                 cluster: WorkloadManager, factory: MagicStateFactory,
                 round_ticks: int, code_distance: int,
                 decode_idle_rounds: bool = False,
                 max_idle_rounds: Optional[int] = None):
        """Wire the chip to its device, controller, cluster, and factory; set the round cadence.
        Each operation's temporal length is NOT a chip parameter: it comes from
        cluster.rounds_for(op) (the ROUNDS policy), so the chip can never disagree with
        the planner/cluster."""
        self.engine = engine
        self.device = device
        self.controller = controller
        self.cluster = cluster
        self.factory = factory
        self.round_ticks = round_ticks
        self.code_distance = code_distance    # patch distance; also the buffer size
        # IDLE-ROUND DECODING (off by default = original trace). arXiv:2511.10633: the
        # stabilization rounds a waiting patch keeps measuring must themselves be decoded
        # (they contribute decoder load and storage error). When True, every commit_rounds()
        # idle rounds the chip submits one memory-window decode job through the SAME
        # DecoderService path the factory uses, so reaction-time waits load the cluster.
        # Known limitation: these ad-hoc memory jobs do not exchange boundaries with
        # each other (the faithful version would plan real scheme windows for the idle
        # period).
        self.decode_idle_rounds = decode_idle_rounds
 
        self.ops: dict[int, Operation] = {}
        # PROGRAM-ORDER RELEASE -- the scheduler. Who runs after who was decided at
        # compile time: op.predecessors lists the earlier ops that share a qubit with
        # it (the "trivial rule" of arXiv:2405.17688 -- for lattice surgery the TRUE
        # constraint). At runtime the chip only counts those down (_release_successors).
        self._deps_remaining: dict[int, int] = {}    # op id -> predecessor bodies not yet finished
        self._op_successors: dict[int, list[int]] = {}  # op id -> ops its body-done releases
        # qubit -> id of the op holding it. A sanity check, NOT the scheduler: starting
        # ops whenever their qubits look free reorders non-commuting gates. See _mark_qubits_busy.
        self.busy_qubits: dict[int, int] = {}
        self.requested: set[int] = set()           # ops whose start sequence has begun
        self.state_ready: set[int] = set()         # ops whose magic state is in hand (or none needed)
        self.started: set[int] = set()
        self.done_bodies: set[int] = set()
        self.gate_released: set[int] = set()       # for non-Clifford gating
        self.last_finish_time = 0
        # safety bound for the continuous idle-round emitter (a gate that never returns would
        # otherwise schedule forever). The default is generous vs any realistic reaction time,
        # but a backlog/divergence study NEEDS waits longer than that -- raise it there, and
        # the chip logs loudly if it ever fires (results past it understate the backlog).
        self.max_idle_rounds = max_idle_rounds if max_idle_rounds is not None \
            else 100 * code_distance
 
    # ---- round cadence ------------------------------------------------------
    def _round_ticks_for(self, op: Operation) -> int:
        """This operation's syndrome-round time. A code may define its own round_us (so
        heterogeneous zones can run different physical cycle times); otherwise the chip's
        global round_ticks applies -- the default, which reproduces the original cadence."""
        rt = getattr(self.cluster.layout.code_for_op(op), "round_us", None)
        return us(rt) if rt is not None else self.round_ticks

    def _round_ticks_for_patch(self, patch) -> int:
        """An idling patch's round time, via ITS code (same fallback as above)."""
        rt = getattr(self.cluster.layout.code_for_patch(patch), "round_us", None)
        return us(rt) if rt is not None else self.round_ticks

    def load(self, ops: list[Operation]) -> None:
        """Register all operations, build the program-order release counters, then
        build the decoder windows."""
        for op in ops:
            self.ops[op.id] = op
            self.cluster.register_op(op)
        for op in ops:
            self._deps_remaining[op.id] = len(op.predecessors)   # bodies this op waits for
            self._op_successors[op.id] = []
        for op in ops:
            for pred in op.predecessors:
                self._op_successors[pred].append(op.id)          # pred's body-done releases op
        self.cluster.build_windows()          # cross-op deps need every op registered first
        # release the ROOTS: an op that waits on nobody starts as soon as the workload loads
        for op in ops:
            if self._deps_remaining[op.id] == 0:
                self._attempt_start(op)

    # ---- starting operations ------------------------------------------------
    def _release_successors(self, op: Operation) -> None:
        """This op's body just finished: every op waiting on it now waits on one fewer
        body, and any successor left waiting on none starts. Only the finishing op's own
        successors are touched -- never the whole workload (Kahn's-algorithm release).
        Only DATA dependencies live here; the two OTHER start conditions -- magic state
        in hand, and (for a gated gate) the prior decode returned -- are waited on later,
        in _maybe_begin, so those waits overlap rather than stack (arXiv:2411.04270)."""
        for succ_id in self._op_successors[op.id]:
            self._deps_remaining[succ_id] -= 1
            if self._deps_remaining[succ_id] == 0:
                self._attempt_start(self.ops[succ_id])

    def _attempt_start(self, op: Operation) -> None:
        """Reserve qubits and fetch the magic state (if any) -- IN PARALLEL with any pending
        reaction. The op physically begins only once BOTH the state is in hand and, if it is a
        gated gate, the prior decode has returned (see _maybe_begin)."""
        self._mark_qubits_busy(op)
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

    def _mark_qubits_busy(self, op: Operation) -> None:
        """Mark this op's qubits busy until its body is done. Dependency release guarantees
        they are free here; a conflict means two ops share a qubit with no ordering edge
        between them (an unwired op list), and silently picking an order would execute a
        different circuit -- so fail loud instead."""
        for q in op.qubits:
            if q in self.busy_qubits:
                if self.busy_qubits[q] == op.id:
                    raise RuntimeError(
                        f"{op.name} lists qubit {q} more than once: {op.qubits}")
                holder = self.ops[self.busy_qubits[q]].name
                raise RuntimeError(
                    f"{op.name} and {holder} share qubit {q} but have no dependency "
                    f"edge -- the operation list is missing program-order wiring "
                    f"(run it through _wire_circuit / a frontend)")
            self.busy_qubits[q] = op.id

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
        self.engine.schedule(self._round_ticks_for(op), lambda: self._round(op, 1),
                             label=f"round1({op.name})")
 
    # ---- per-round cadence --------------------------------------------------
    def _round(self, op: Operation, r: int) -> None:
        """Emit one syndrome round through the controller to the decoder cluster."""
        total = self.cluster.rounds_for(op)          # this op's length via the ROUNDS policy (D1)
        # PER-PATCH seam: a device may emit one payload PER PATCH for this round (the
        # optional round_payloads hook); the default single-payload contract is unchanged.
        # Each fragment is tagged with the fragment count so the cluster only counts the
        # round as arrived once all of them are in.
        emit = getattr(self.device, "round_payloads", None)
        payloads = emit(op, r) if emit is not None else [self.device.round_payload(op, r)]
        self.engine.log("Chip", f"{op.name} fires round {r}/{total}")
        # hand the round's syndromes to the controller, which relays them to the decoder
        for payload in payloads:
            payload.n_fragments = len(payloads)
            self.controller.relay_syndrome(payload, self.cluster.on_syndrome_arrival)
        if r < total:
            self.engine.schedule(self._round_ticks_for(op), lambda: self._round(op, r + 1),
                                 label=f"round{r+1}({op.name})")
        else:
            self._body_done(op)
 
    def _body_done(self, op: Operation) -> None:
        """Op's physical work done; start successors and emit idle rounds while a gated successor waits."""
        self.done_bodies.add(op.id)
        self.last_finish_time = max(self.last_finish_time, self.engine.now)
        self.engine.log("Chip", f"{op.name} BODY DONE")
        for q in op.qubits:
            del self.busy_qubits[q]            # freed FIRST: a successor released below may need them
        self._release_successors(op)
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
        # gate returns in on_decision) or has started. Idle rounds are routed via
        # on_memory_round (a count that fills window buffers); whether they are ALSO decoded
        # is the decode_idle_rounds flag set in __init__ -- arXiv:2511.10633 says these memory
        # stabilization rounds do require decoding (they contribute decoder load and storage
        # error), so leaving the flag off slightly UNDERSTATES decoder utilization during
        # reaction waits (the byte-identical default; see __init__ for the trade-off).
        if op.has_successor and self._has_waiting_gated_successor(op.id):
            self.engine.log("Chip",
                            f"{op.name} patch idles (successor gated on a decode); "
                            f"emitting memory rounds every round until the correction returns")
            patch = op.patches[0] if op.patches else (op.qubits[0] if op.qubits else 0)
            self.engine.schedule(self._round_ticks_for_patch(patch),
                                 lambda oid=op.id, pt=patch: self._emit_idle_round(oid, pt, 1),
                                 label=f"idle-tick({op.name},1)")
 
    def _has_waiting_gated_successor(self, op_id: int) -> bool:
        """True while a successor of this op is still blocked on a decode result: it is
        gated, its gate has not been released, and it has not started. Checks only this
        op's own successors, never the whole workload."""
        for succ_id in self._op_successors[op_id]:
            succ = self.ops[succ_id]
            if (succ.gated_by is not None and succ.id not in self.gate_released
                    and succ.id not in self.started):
                return True
        return False
 
    def _emit_idle_round(self, op_id: int, patch, k: int) -> None:
        """Emit idle memory round k for an idling patch, then schedule the next one -- continuing
        every round until the gated successor is released (or has started), modelling a QPU that
        keeps measuring stabilizers while it waits in storage. Self-terminating + capped."""
        if not self._has_waiting_gated_successor(op_id):
            return                                          # gate returned (or started): stop
        if k > self.max_idle_rounds:
            self.engine.log("Chip",
                            f"WARNING: {self.ops[op_id].name} hit the idle-round cap "
                            f"(max_idle_rounds={self.max_idle_rounds}) with its gated "
                            f"successor still waiting -- no more memory rounds will be "
                            f"emitted, so decoder load and backlog past this point are "
                            f"UNDERSTATED. Raise max_idle_rounds for long-reaction studies.")
            return
        self.controller.relay_syndrome(
            SyndromePayload(op_id, patch, k),
            lambda p, o=op_id: self.cluster.on_memory_round(o))
        if self.decode_idle_rounds:
            # every commit-region's worth of idle rounds becomes one memory-window decode
            # job (commit + buffer rounds), sized to this patch's code -- the decoder load
            # of waiting in storage (arXiv:2511.10633).
            code = self.cluster.layout.code_for_patch(patch)
            if k % code.commit_rounds() == 0:
                self.cluster.submit_decode(
                    code.commit_rounds() + code.buffer_rounds(),
                    on_done=lambda: None, code=code.name,
                    spatial_nodes=code.spatial_nodes(1),
                    label=f"mem({self.ops[op_id].name},r{k})")
        self.engine.schedule(self._round_ticks_for_patch(patch),
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
 