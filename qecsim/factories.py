# =========================================================================
# FACTORIES
# This module holds magic state supply factories, where every non-clifford 
# operation gets its magic states from.
# =========================================================================


from __future__ import annotations
 
from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING
 
from .config import fmt
from .engine import Engine

if TYPE_CHECKING:
    from .protocols import DecodeService


# TODO: This is for testing need to fully verify and update it so that its fully realistic
class InfiniteFactory:
    """Simple factory that always has a magic state in stock)."""
    def __init__(self, engine: Engine):
        """Just keep a handle to the engine."""
        self.engine = engine
 
    def request(self, op_id: int, callback: Callable[[], None]) -> None:
        """Deliver instantly. NOTE: IDEALIZED UNLIMITED supply (no distillation modeled)."""
        callback()
 
    def shutdown(self) -> None:
        """Nothing to stop."""
        pass

# TODO: This is for testing need to fully verify and update it so that its fully realistic
class DistillationFactory:
    """
    A 15-to-1 magic state factory, designed based on the paper's behaviour.
 
      - `num_units` parallel distillation units run continuously in the background.
      - Each unit performs an attempt every `cycle_ticks` (= 11 * tau_logical, the
        11 commuting rotations of the 15-to-1 protocol).
      - On success (probability `p_success`) the unit submits `n_corr` correction-qubit
        decode jobs to the REAL decoder cluster via `decode_service`. Because the 15-to-1
        rotations COMMUTE, these jobs are independent and decode in PARALLEL (no chain) --
        but they compete for the same decoder units as the core, which is why the paper
        budgets ~10% extra decoders for the factory. When all of a state's correction
        jobs finish, one return trip (`return_ticks`) later the state enters the store.
        The hold the magic state experiences is therefore EMERGENT (queue wait + parallel
        decode + return), not a fixed input.
      - On failure the attempt is discarded and retried (the distillation discard rate).
      - `request` is served from the store FIFO; an empty store STALLS the requester
        (the supply stall).
    """
    def __init__(self, engine: Engine, num_units: int, cycle_ticks: int,
                 decode_service: "DecodeService", corr_rounds: int, n_corr: int = 11,
                 return_ticks: int = 0, p_success: float = 1.0, seed: int = 0,
                 initial_store: int = 0):
        """Set up a single-level factory: units, cycle time, success rate, store."""
        import random
        self.engine = engine
        self.num_units = num_units
        self.cycle_ticks = cycle_ticks
        self.decode_service = decode_service
        self.corr_rounds = corr_rounds
        self.n_corr = n_corr
        self.return_ticks = return_ticks
        self.p_success = p_success
        self.rng = random.Random(seed)
 
        self.store = initial_store             # warm start: states already in stock
        self.waiting: list[tuple[int, Callable[[], None]]] = []
        self.produced = 0
        self.in_flight = 0                     # states distilled, correction-decoding
        self.busy_units = 0                    # units currently distilling
        self.peak_in_flight = 0
        self.total_stall = 0
        self._stall_start: dict[int, int] = {}
        self._shutdown = False
        # NOTE: production is DEMAND-DRIVEN. Units are launched by _maybe_start() only
        # when there is an unserved request; the factory does not free-run on a timer.
 
    def shutdown(self) -> None:
        """Stop launching new attempts (called when the circuit is complete)."""
        self._shutdown = True
 
    def _maybe_start(self) -> None:
        """Launch distillation attempts only while there is unmet demand -- a waiting
        request not already covered by a state currently being distilled or decoded.
        Idle units stay idle; the factory never produces states nobody asked for."""
        while (not self._shutdown
               and self.busy_units < self.num_units
               and len(self.waiting) > self.busy_units + self.in_flight):
            self.busy_units += 1
            self.engine.schedule(self.cycle_ticks, self._attempt_done,
                                 label="distill_attempt")
 
    def _attempt_done(self) -> None:
        """A distillation attempt finished; on success, queue its correction decode."""
        self.busy_units -= 1
        if self.rng.random() < self.p_success:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            remaining = {"n": self.n_corr}
            self.engine.log("Factory",
                            f"a unit distilled a state; submitting {self.n_corr} "
                            f"correction-qubit decode jobs to the cluster (parallel)")
            for _ in range(self.n_corr):
                self.decode_service.submit_decode(
                    self.corr_rounds,
                    on_done=lambda rem=remaining: self._corr_done(rem),
                    label="MSF-corr")
        else:
            self.engine.log("Factory", "a unit's distillation DISCARDED, retrying")
        self._maybe_start()                    # keep going only if demand remains
 
    def _corr_done(self, remaining: dict) -> None:
        """A correction decode finished; the state is now ready in the store."""
        remaining["n"] -= 1
        if remaining["n"] == 0:
            self.engine.schedule(self.return_ticks, self._release,
                                 label="distill_release")
 
    def _release(self) -> None:
        """Hand a finished state to the oldest waiting request."""
        self.in_flight -= 1
        self.store += 1
        self.produced += 1
        self.engine.log("Factory", f"magic state ready (store now {self.store})")
        self._fulfil()
        self._maybe_start()
 
    def request(self, op_id: int, callback: Callable[[], None]) -> None:
        """A gate asks for a state: deliver now if in stock, else deliver when ready."""
        self.waiting.append((op_id, callback))
        self._stall_start[op_id] = self.engine.now
        self.engine.log("Factory",
                        f"op#{op_id} requests a magic state "
                        f"(store {self.store}, waiting {len(self.waiting)})")
        self._fulfil()
        self._maybe_start()
 
    def _fulfil(self) -> None:
        """Deliver a state to a waiting request and log it."""
        while self.store > 0 and self.waiting:
            self.store -= 1
            op_id, cb = self.waiting.pop(0)
            waited = self.engine.now - self._stall_start.pop(op_id, self.engine.now)
            self.total_stall += waited
            tag = "" if waited == 0 else f"  (supply stall {fmt(waited).strip()})"
            self.engine.log("Factory",
                            f"  -> delivered to op#{op_id} (store now {self.store}){tag}")
            cb()

@dataclass
class DistillLevel:
    """One distillation level of a multi-level MSF (Silva et al., arXiv:2411.04270, II B)."""
    units: int            # u_l : parallel distillation units at this level
    d: int                # code distance d_l at this level (increasing up the chain)
    O: int = 13           # logical cycles per distillation round (13 first level, 15 higher)
    P: float = 1.0        # success probability of a distillation round at this level
 
# TODO: This is for testing need to fully verify and update it so that its fully realistic
class MultiLevelDistillationFactory:
    """A MULTI-LEVEL magic state factory: a supply chain of distillation levels feeding the
    core, faithful to Silva et al., "Optimizing Multi-level Magic State Factories for
    Fault-Tolerant Quantum Architectures" (arXiv:2411.04270).
 
    Pipeline (level 0 = preparation, levels 1..L = distillation, L+1 = core consumer):
      - Level 0: each preparation unit injects a physical magic state into a distance-d0
        patch, producing one prepared state every prep_O*d0 rounds with success prep_P.
      - Level l (1..L): each unit consumes M lower-level states from the buffer below and,
        after O_l*d_l*W ticks, produces N higher-fidelity states with success probability P
        (failure discards the M inputs and retries). For 15:1: M=15, N=1, O=13/15.
      - Buffers between levels (self.buffer[l]) hold produced states so a unit can begin its
        next round without waiting for downstream consumption -- the paper's buffer registers
        that keep the chain in long-term steady state.
      - The top level L feeds the core via request(); ONE final state costs M^L prepared
        states cascading up the chain (e.g. 225 prepared states for L=2, M=15).
 
    Production is DEMAND-DRIVEN (pull): a core request propagates demand down the chain via
    the rate relations D_{l-1} >= C_l (paper Eqs 6-9); each level distills only what is needed
    downstream and inputs allow, so nothing free-runs. The post-corrected protocol's
    correction-qubit decoding is routed through the decoder cluster (DecodeService), so the
    factory's classical load competes with the core for decoder units.
    """
    def __init__(self, engine: Engine, levels: list[DistillLevel], *,
                 W_ticks: int, M: int = 15, N: int = 1,
                 prep_units: int = 1, prep_O: int = 2, prep_d: int = 3, prep_P: float = 1.0,
                 decode_service: Optional["DecodeService"] = None,
                 corr_rounds: int = 0, n_corr: int = 0, seed: int = 0):
        """Set up the multi-level supply chain (prep -> levels -> core)."""
        import random
        self.engine = engine
        self.levels = levels                       # levels[0] is level 1, ... levels[L-1] is level L
        self.L = len(levels)
        self.M = M
        self.N = N
        self.W = W_ticks                           # per-round (parity-check) time
        self.prep_units = prep_units
        self.prep_time = prep_O * prep_d * W_ticks
        self.prep_P = prep_P
        self.decode_service = decode_service
        self.corr_rounds = corr_rounds
        self.n_corr = n_corr
        self.rng = random.Random(seed)
 
        # round time per distillation level l in {1..L}
        self.round_time = {l: levels[l - 1].O * levels[l - 1].d * W_ticks
                           for l in range(1, self.L + 1)}
        # buffers[l] for l in 0..L : buffer[0]=prepared states, buffer[L]=final states
        self.buffer = {l: 0 for l in range(0, self.L + 1)}
        self.busy = {l: 0 for l in range(0, self.L + 1)}     # units mid-round per level
        self.produced = {l: 0 for l in range(0, self.L + 1)}
        self.failures = {l: 0 for l in range(0, self.L + 1)}
 
        self.waiting: list[tuple[int, Callable[[], None]]] = []
        self.total_stall = 0
        self._stall_start: dict[int, int] = {}
        self.peak_in_flight = 0
        self._shutdown = False
 
    def shutdown(self) -> None:
        """Stop the production loop."""
        self._shutdown = True
 
    # ---- the consumer interface (drop-in for MagicStateFactory) -------------
    def request(self, op_id: int, callback: Callable[[], None]) -> None:
        """A gate asks for a final state; record demand and start producing."""
        self.waiting.append((op_id, callback))
        self._stall_start[op_id] = self.engine.now
        self.engine.log("Factory",
                        f"op#{op_id} requests a magic state "
                        f"(top-level store {self.buffer[self.L]}, waiting {len(self.waiting)})")
        self._drive()
 
    def _fulfil_core(self) -> None:
        """Deliver a finished final state to a waiting request."""
        while self.buffer[self.L] > 0 and self.waiting:
            self.buffer[self.L] -= 1
            op_id, cb = self.waiting.pop(0)
            waited = self.engine.now - self._stall_start.pop(op_id, self.engine.now)
            self.total_stall += waited
            tag = "" if waited == 0 else f"  (supply stall {fmt(waited).strip()})"
            self.engine.log("Factory", f"  -> delivered final state to op#{op_id}{tag}")
            cb()
 
    # ---- the pull engine: propagate demand down, start rounds where possible ----
    def _drive(self) -> None:
        """Pull engine: recompute demand top-down each tick and start the work each level can do."""
        if self._shutdown:
            return
        import math
        L, M, N = self.L, self.M, self.N
        self._fulfil_core()
        # Pull loop: each iteration recomputes demand top-down from the CURRENT buffers and
        # in-flight rounds (so rounds already started are not re-counted), then starts what it
        # can. Recomputing inside the loop is what prevents over-production.
        progress = True
        while progress:
            progress = False
            need = {L: len(self.waiting)}
            for l in range(L, 0, -1):
                deficit = max(0, need[l] - self.buffer[l] - self.busy[l] * N)
                rounds = math.ceil(deficit / N) if deficit > 0 else 0
                need[l - 1] = M * rounds          # each level-l round consumes M level-(l-1)
            # preparation (level 0)
            idle0 = self.prep_units - self.busy[0]
            deficit0 = max(0, need[0] - self.buffer[0] - self.busy[0])
            while deficit0 > 0 and idle0 > 0:
                self.busy[0] += 1
                self.engine.schedule(self.prep_time, self._prep_done, label="prep")
                idle0 -= 1; deficit0 -= 1; progress = True
            # distillation levels 1..L
            for l in range(1, L + 1):
                deficit = max(0, need[l] - self.buffer[l] - self.busy[l] * N)
                rounds_wanted = math.ceil(deficit / N) if deficit > 0 else 0
                idle = self.levels[l - 1].units - self.busy[l]
                while rounds_wanted > 0 and idle > 0 and self.buffer[l - 1] >= M:
                    self.buffer[l - 1] -= M               # consume M inputs from below
                    self.busy[l] += 1
                    self._start_round(l)                  # waits for BOTH time AND decoding
                    idle -= 1; rounds_wanted -= 1; progress = True
        self.peak_in_flight = max(self.peak_in_flight, sum(self.busy.values()))
 
    def _start_round(self, l: int) -> None:
        """Begin a level-l distillation round. Its produced state becomes available only when
        BOTH complete: (a) the physical distillation time round_time[l], and (b) ALL n_corr
        correction-qubit decodes routed through the shared decoder cluster. Tying the state to
        the decode is what exposes the factory to decoder contention -- without it, the chain
        advances as if classical decoding were free (immune to a swamped cluster)."""
        rd = {"l": l, "phys": False, "decodes_left": 0, "done": False}
        # (b) correction-qubit decodes share the decoder units (DecodeService)
        if self.decode_service is not None and self.n_corr:
            rd["decodes_left"] = self.n_corr
            for _ in range(self.n_corr):
                self.decode_service.submit_decode(
                    self.corr_rounds, on_done=lambda r=rd: self._corr_done(r),
                    label=f"MSF-corr-L{l}")
        # (a) physical distillation time
        self.engine.schedule(self.round_time[l], lambda r=rd: self._phys_done(r),
                             label=f"distill_L{l}")
 
    def _phys_done(self, rd: dict) -> None:
        """The physical distillation time elapsed; finish the round if decoding is also done."""
        rd["phys"] = True
        self._finish_round(rd)
 
    def _corr_done(self, rd: dict) -> None:
        """One correction decode came back; finish the round once all of them AND the time are in."""
        rd["decodes_left"] -= 1
        self._finish_round(rd)
 
    def _finish_round(self, rd: dict) -> None:
        """Produce the state IFF both the physical time and every correction decode are complete."""
        if rd["done"] or not rd["phys"] or rd["decodes_left"] > 0:
            return
        rd["done"] = True
        l = rd["l"]
        self.busy[l] -= 1                                 # the unit is free only now (time+decode)
        if self.rng.random() < self.levels[l - 1].P:
            self.buffer[l] += self.N
            self.produced[l] += self.N
            where = "FINAL state -> core buffer" if l == self.L else f"level-{l} state -> buffer"
            self.engine.log("Factory",
                            f"level {l} distilled a state ({where}; "
                            f"consumed {self.M} level-{l-1} states)")
        else:
            self.failures[l] += 1
            self.engine.log("Factory", f"level {l} distillation FAILED (inputs discarded), retrying")
        self._drive()
 
    def _prep_done(self) -> None:
        """A level-0 prepared state is ready; add it to the buffer."""
        self.busy[0] -= 1
        if self.rng.random() < self.prep_P:
            self.buffer[0] += 1
            self.produced[0] += 1
        else:
            self.failures[0] += 1
        self._drive()
  