from __future__ import annotations
 
from typing import Callable, Optional, TYPE_CHECKING
 
from .config import fmt
from .engine import Engine
from .message import Operation, SyndromePayload, DecodeJob, DecodeResult, Window, WindowPlan
 
from .codes import SurfaceCodeModel
from .schemes import SlidingWindowScheme
from .planner import WindowPlanner, FixedRounds
from .layouts import UniformLayout
 
if TYPE_CHECKING:                      # type-only; these collaborators are handed to __init__
    from .protocols import (Decoder, Scheduler, Controller, Orchestrator,
                            CodeModel, DecodingScheme, LayoutModel, RoundsPolicy)
# ========================================================================================
# CLUSTER
# This module define the decoder cluster, the papers window manger and the main peice of 
# the decoding side.  
# It receives the WindowPlan from the orchestrator's planner ahead of time (load_execution_plan),
# runs the queue against it, exchanges committed window boundaries between decoders (the t_dd hop),
# and DELIVERS each finished operation result to the orchestrator (the t_do hop) where, after
# the sending the decode, the ORCHESTRATOR owns sending any conditional instruction back to the chip.
# The cluster never reaches into the chip or factory (dependency inversion); it only fires the
# lifecycle callback `on_workload_complete` the wiring installs.
# ========================================================================================

class DecoderCluster:
    """
    Owns three things:
      - per-operation window buffers (rounds accumulate here until complete),
      - the BLOCKED set (window complete but a dependency unmet -> not runnable),
      - the READY QUEUE + a pool of decoder units (runnable jobs wait for a free unit).
        The ready queue is THE queue of syndromes/jobs waiting to be decoded.
    """

    def __init__(self, engine: Engine, decoder: Decoder, scheduler: Scheduler,
                 controller: Controller, orchestrator: Orchestrator,
                 num_units: int, code_distance: Optional[int] = None,
                 rounds_per_op: int = 11, *, code: Optional[CodeModel] = None,
                 scheme: Optional[DecodingScheme] = None,
                 layout: Optional[LayoutModel] = None,
                 decoders: Optional[dict] = None,
                 rounds_policy: Optional[RoundsPolicy] = None):
        """Set up the cluster: decoder units, ready queue, syndrome buffer, bookkeeping."""
        self.engine = engine
        self.decoder = decoder
        # G1 -- HETEROGENEOUS PER-CODE DECODING. `decoder` is the DEFAULT decoder, used for any
        # job whose code is not in `decoders`. `decoders` is an optional {code_name: Decoder}
        # map so each code can be decoded by its own algorithm (surface by MWPM, BB by Relay-BP,
        # ...). A job carries its code on DecodeJob.code; _decoder_for(code) picks the right one.
        # With no map (the common single-code case) every job routes to `decoder`, so behaviour
        # is byte-identical to before this seam existed.
        self.decoders = dict(decoders) if decoders else {}
        self.scheduler = scheduler
        self.controller = controller
        self.orchestrator = orchestrator
        self.num_units = num_units
        self.free_units = num_units
        # The QEC code is now a swappable model. Passing code_distance=d (back-compatible)
        # constructs a surface code of that distance, reproducing the original behavior.
        if code is None and layout is None:
            if code_distance is None:
                raise ValueError("provide code=<CodeModel>, layout=<LayoutModel>, "
                                 "or code_distance=<int>")
            code = SurfaceCodeModel(d=code_distance)
        # The LAYOUT seam: which code each patch uses. A single code (or code_distance) means a
        # UniformLayout -- one zone -- which reproduces the original single-code behavior exactly.
        # A heterogeneous QPU passes layout=ZonedLayout(...) instead.
        if layout is None:
            self.layout = UniformLayout(code)
        else:
            self.layout = layout
        # A representative code for the summary log and the window-timing defaults below. For a
        # uniform layout this IS the one code, so the logged plan is unchanged.
        self.code = code if code is not None else self.layout.codes()[0]
        # The window/commit policy is a swappable scheme; the default is fixed sliding window.
        self.scheme = scheme if scheme is not None else SlidingWindowScheme()
        self.d = self.code.distance
        self.commit = self.code.commit_rounds()     # commit region (d rounds for surface code)
        self.buffer = self.code.buffer_rounds()     # buffer/lookahead (d rounds for surface code)
        # ROUNDS seam (D1 fix). rounds-per-operation is a swappable policy. A plain rounds_per_op
        # int is wrapped as FixedRounds (the original behaviour, byte-identical regression); pass
        # rounds_policy=CodeRounds() for per-code rounds so high-distance zones' buffers fit.
        self.rounds_policy = rounds_policy if rounds_policy is not None \
            else FixedRounds(rounds_per_op)
        # representative count for the summary log and any external readers; per-op work uses
        # self.rounds_for(op). For FixedRounds this equals the old global rounds_per_op.
        self.R = self.rounds_policy.rounds_for(None, self.code)
 
        # per-op bookkeeping
        self.ops: dict[int, Operation] = {}
        self.rounds_arrived: dict[int, int] = {}   # highest round number arrived for op
        self.memory_rounds: dict[int, int] = {}    # idle memory rounds (gated successors)
        # REAL syndrome buffer: arriving payloads are retained here (the decoder-cluster RAM
        # of arXiv:2511.10633 Sec III) until their window decodes, then assembled into the
        # decode job. Keyed [op_id][round_index] -> SyndromePayload. In timing-only mode the
        # payloads carry no bits, but the plumbing is identical, so a real decoder needs no
        # cluster change -- only a real DeviceModel + Decoder.
        self.payload_store: dict[int, dict[int, SyndromePayload]] = {}
        self.peak_payloads = 0                     # peak retained payloads (storage high-water)
        self.num_ops = 0
 
        # the sliding-window plan (built once all ops are loaded)
        self.windows: dict[tuple, Window] = {}
        self.op_windows: dict[int, list] = {}
        self.nwin: dict[int, int] = {}
        self.successors: dict[int, list] = {}
        self.committed_windows: set = set()
        self.op_results: dict[int, int] = {}      # accumulated logical value per op (real decoders)
        self.total_windows = 0
        self._windows_built = False
        self._plan_spatial = None                 # per-op decode sizes from the loaded plan
        self.ready: list[DecodeJob] = []          # THE ready queue
        self.queue_log: list[tuple[int, int]] = []
        # Dependency INVERSION (no back-reference to the chip): the cluster -- the paper's
        # "workload manager" (arXiv:2511.10633 Sec III) -- never reaches into the chip or factory.
        # It queues decoding jobs, exchanges committed window boundaries (the t_dd dependencies),
        # and DELIVERS each finished result to the orchestrator (the t_do hop). It does NOT own
        # the conditional return path: dispatching a decision back to the chip is the
        # ORCHESTRATOR's job (orchestrator.integrate -> controller -> chip). The cluster keeps one
        # lifecycle sink the wiring fills in:
        #   on_workload_complete()  -- fire when the last window has committed (e.g. factory.shutdown)
        self.on_workload_complete = None          # Optional[Callable[[], None]]
 
    def register_op(self, op: Operation) -> None:
        """Start tracking a new operation (arrived rounds, payload buffer, etc.).
        Idempotent: re-registering an already-known op refreshes the reference but keeps its
        runtime state, so the orchestrator can register ops before the chip loads them."""
        if op.id not in self.ops:
            self.rounds_arrived[op.id] = 0
            self.memory_rounds[op.id] = 0
            self.payload_store[op.id] = {}
        self.ops[op.id] = op
        self.num_ops = len(self.ops)

    def _decoder_for(self, code: Optional[str]) -> "Decoder":
        """Pick the decoder for a job's code (G1). Falls back to the default `decoder` when the
        code has no dedicated entry -- so a single-code run always uses the one default decoder."""
        return self.decoders.get(code, self.decoder)
 
    def rounds_for(self, op: Operation) -> int:
        """Rounds this operation runs for, via the ROUNDS policy under the op's own code (D1).
        The planner, this cluster (window completion / payload assembly), and the chip all read
        this, so they agree on each operation's temporal length even across heterogeneous codes."""
        return self.rounds_policy.rounds_for(op, self.layout.code_for_op(op))
 
    def _spatial_nodes(self, op: Operation) -> int:
        # The decode-job size is part of the plan the orchestrator hands over; if a plan is
        # loaded, read it from there. Otherwise (no plan yet) fall back to the layout directly.
        # A uniform layout reproduces the single code's aggregate spatial_nodes(num_patches).
        """Decoding-graph size for an operation (from the loaded plan, else the layout)."""
        if self._plan_spatial is not None and op.id in self._plan_spatial:
            return self._plan_spatial[op.id]
        return self.layout.spatial_nodes_for(op)
 
    def load_execution_plan(self, plan: WindowPlan) -> None:
        """Receive the pre-computed WindowPlan from the orchestrator's planner and install it
        (arXiv:2511.10633 Sec III: the plan is communicated to the cluster AHEAD OF TIME). The
        cluster does NOT compute windows in this path -- it just holds the plan and runs the
        queue against it at runtime. Called once, before syndromes flow."""
        if self._windows_built:
            return
        self._windows_built = True
        self.windows = plan.windows
        self.nwin = plan.nwin
        self.op_windows = plan.op_windows
        self.successors = plan.successors
        self._plan_spatial = plan.spatial_nodes
        self.total_windows = plan.total_windows
        # wire dependents into the cluster's view (the plan already filled them on the Windows)
        self.engine.log("DecoderClstr",
                        f"received execution plan: d={self.d}, commit={self.commit}, "
                        f"buffer={self.buffer}, {self.R} rounds/op -> "
                        f"{plan.nwin.get(next(iter(self.ops), 0), 0)} "
                        f"windows per operation, {plan.total_windows} windows total")
 
    def build_windows(self) -> None:
        """Back-compat entry: if no execution plan has been loaded yet, build one in place
        (using this cluster's scheme + layout) and install it.
 
        The PREFERRED path is for the orchestrator's WindowPlanner to compute the plan and call
        load_execution_plan() ahead of time -- the planning then lives in the planner, not here
        (see the ROLE MAP in the Section 3 banner). This shim keeps direct chip.load() callers
        working and produces an identical plan, so behavior is unchanged."""
        if self._windows_built:
            return
        planner = WindowPlanner(self.scheme, self.layout, self.rounds_policy)
        self.load_execution_plan(planner.plan(list(self.ops.values())))
 
    # ---- a syndrome round has arrived from the controller -------------------
    def on_syndrome_arrival(self, payload: SyndromePayload) -> None:
        """A syndrome round arrived: buffer it and re-check which windows can now decode."""
        op = self.ops[payload.operation_id]
        self.rounds_arrived[op.id] = max(self.rounds_arrived[op.id], payload.round_index)
        # retain the payload in the cluster's syndrome buffer until its window decodes
        self.payload_store[op.id][payload.round_index] = payload
        self.peak_payloads = max(self.peak_payloads,
                                 sum(len(s) for s in self.payload_store.values()))
        self.engine.log("DecoderClstr",
                        f"round {payload.round_index} of {op.name} arrived "
                        f"(op now has rounds 1..{self.rounds_arrived[op.id]})")
        for k in range(self.nwin[op.id]):
            self._check_window((op.id, k))
        # this op's early rounds also feed the BUFFER overflow of its predecessors' last windows
        for pred_id in op.predecessors:
            for k in range(self.nwin[pred_id]):
                self._check_window((pred_id, k))
 
    def on_memory_round(self, op_id: int) -> None:
        """An idle/memory round arrived for a waiting patch (fills window buffers)."""
        self.memory_rounds[op_id] += 1
        self.engine.log("DecoderClstr",
                        f"memory round for {self.ops[op_id].name} "
                        f"(idle buffer rounds: {self.memory_rounds[op_id]})")
        for k in range(self.nwin[op_id]):
            self._check_window((op_id, k))
 
    def _assemble_payloads(self, w: Window) -> list:
        """Collect retained payloads for this window's commit+buffer rounds (plus the
        successor-operation overflow rounds), so a real decoder receives the actual data.
        Timing-only payloads carry no bits, but the assembly path is identical."""
        op_store = self.payload_store.get(w.op_id, {})
        R_op = self.rounds_for(self.ops[w.op_id])      # this operation's own length (D1)
        hi = min(w.buffer_hi, R_op)
        out = [op_store[r] for r in range(w.commit_lo, hi + 1) if r in op_store]
        overflow = w.buffer_hi - R_op
        if overflow > 0:
            for s in self.successors.get(w.op_id, []):
                succ = self.payload_store.get(s, {})
                out += [succ[r] for r in range(1, overflow + 1) if r in succ]
        return out
 
    def _window_data_complete(self, w: Window) -> bool:
        """Whether a window has all the rounds it needs (delegates to the scheme)."""
        op = self.ops[w.op_id]
        succ_rounds = max((self.rounds_arrived[s] for s in self.successors[w.op_id]),
                          default=0)
        return self.scheme.data_complete(
            w, rounds_arrived=self.rounds_arrived[w.op_id], successor_rounds=succ_rounds,
            memory_rounds=self.memory_rounds[w.op_id], n_rounds=self.rounds_for(op),
            has_successor=op.has_successor, op=op, layout=self.layout)
 
    def _check_window(self, key: tuple) -> None:
        """If a window has its data and its dependencies, build its job and enqueue it."""
        w = self.windows[key]
        if w.queued or w.committed:
            return
        if not self._window_data_complete(w):
            return
        op = self.ops[w.op_id]
        if w.deps_remaining > 0:
            if not w.blocked_logged:
                w.blocked_logged = True
                self.engine.log("DecoderClstr",
                                f"{op.name} W{w.k} (commit {w.commit_lo}-{w.commit_hi}) "
                                f"has all its data, but is WAITING for the boundary from "
                                f"{w.deps_remaining} predecessor window(s)")
            return
        job = DecodeJob(op_id=w.op_id, window_id=w.k, n_rounds=w.n_rounds,
                        ready_time=self.engine.now, deadline=self.engine.now,
                        spatial_nodes=self._spatial_nodes(op),
                        payloads=self._assemble_payloads(w),
                        code=self.layout.code_for_op(op).name,   # G1: route to this code's decoder
                        label=f"{op.name} W{w.k}[commit {w.commit_lo}-{w.commit_hi}]")
        self.scheduler.insert(self.ready, job)
        w.queued = True
        self.engine.log("DecoderClstr",
                        f"{op.name} W{w.k} (commit {w.commit_lo}-{w.commit_hi}) READY "
                        f"-> enqueue (ready-queue length = {len(self.ready)})")
        self.queue_log.append((self.engine.now, len(self.ready)))
        self._try_dispatch()
 
    def submit_decode(self, n_rounds: int, on_done: Callable[[], None],
                      label: str = "external", deadline: Optional[int] = None,
                      code: Optional[str] = None) -> None:
        """DecodeService entry point: submit a self-contained decode job that competes for
        the SAME decoder units as the core (used by the magic state factory). `code` (optional)
        routes the job to that code's decoder; None uses the cluster's default decoder (G1)."""
        job = DecodeJob(op_id=-1, window_id=0, n_rounds=n_rounds,
                        ready_time=self.engine.now,
                        deadline=self.engine.now if deadline is None else deadline,
                        on_done=on_done, label=label, code=code)
        self.scheduler.insert(self.ready, job)
        self.queue_log.append((self.engine.now, len(self.ready)))
        self._try_dispatch()
 
    def _try_dispatch(self) -> None:
        """While a decoder unit is free, pop the next ready job and run its decode latency."""
        while self.free_units > 0 and self.ready:
            job = self.scheduler.pop(self.ready)
            self.free_units -= 1
            lat = self._decoder_for(job.code).latency(job)   # G1: per-code decoder
            waited = self.engine.now - job.ready_time
            self.engine.log("DecoderClstr",
                            f"START DECODE {job.label} (waited {fmt(waited).strip()} in queue, "
                            f"units free now {self.free_units})")
            self.queue_log.append((self.engine.now, len(self.ready)))
            self.engine.schedule(lat, lambda j=job: self._on_decode_done(j),
                                 label=f"decode_done({job.label})")
 
    def _on_decode_done(self, job: DecodeJob) -> None:
        """Commit a finished window, hand its boundary to the next window, report the op result on its last window."""
        self.free_units += 1
        if job.on_done is not None:                  # factory correction-qubit job
            self.engine.log("DecoderClstr",
                            f"DECODE DONE {job.label} (units free now {self.free_units})")
            job.on_done()
            self._try_dispatch()
            return
        # ---- an operation window ----
        key = (job.op_id, job.window_id)
        w = self.windows[key]
        op_id = job.op_id
        op = self.ops[op_id]
        w.committed = True
        self.committed_windows.add(key)
        self.engine.log("DecoderClstr",
                        f"DECODE DONE {op.name} W{w.k}: rounds {w.commit_lo}-{w.commit_hi} "
                        f"committed (units free now {self.free_units})")
        # A committed window passes its boundary information to the next window's decoder.
        # This is a decoder->decoder exchange: SENT now, ARRIVES one t_dd hop later.
        for dep_key in w.dependents:
            dst = self.ops[dep_key[0]]
            self.engine.log("DecoderClstr",
                            f"decoder->decoder SEND: {op.name} W{w.k} -> {dst.name} "
                            f"W{dep_key[1]}  (boundary/artificial defects, arrives in t_dd)")
            self.engine.schedule(self.controller.dec_to_dec_delay(),
                                 lambda dk=dep_key, sn=op.name, sk=w.k:
                                     self._receive_boundary(dk, sn, sk),
                                 label=f"defects {op.name}W{w.k}->{dst.name}W{dep_key[1]}")
        # run the actual decode and KEEP its result. For a real decoder this is a genuine
        # logical value; for a timing-only stub it is None and the orchestrator falls back
        # to its toy outcome. Per-operation windows are combined (parity) into one outcome.
        res = self._decoder_for(job.code).decode(job)        # G1: per-code decoder
        if res is not None and res.logical_value is not None:
            self.op_results[op_id] = self.op_results.get(op_id, 0) ^ int(res.logical_value)
        # the operation's logical outcome is known when its LAST window commits -> orchestrator
        if w.k == self.nwin[op_id] - 1:
            self.engine.schedule(self.controller.dec_to_orch_delay(),
                                 lambda: self._deliver_to_orchestrator(op),
                                 label=f"result->orch({op.name})")
            # RELEASE this op's syndrome RAM now its last window has decoded (arXiv:2511.10633
            # Sec III: the decoder-cluster RAM holds syndromes only while their window is being
            # decoded, and that storage is itself a headline cost in the paper). Safe to free here:
            # all of this op's own windows are committed, and any predecessor whose buffer
            # overflowed into this op's early rounds already read them -- that predecessor's last
            # window had to commit before this op's window 0 could (the window dependency). Without
            # this the store grows monotonically and peak_payloads stops being a real high-water.
            self.payload_store.pop(op_id, None)
        self._try_dispatch()
        # WORKLOAD COMPLETE: the last window has committed. The cluster does not know about
        # factories or chips -- it just fires the lifecycle callback the wiring installed.
        if len(self.committed_windows) == self.total_windows and self.on_workload_complete is not None:
            self.on_workload_complete()
 
    def _receive_boundary(self, key: tuple, src_name: str, src_k: int) -> None:
        """A neighbor window's boundary arrived; clear one dependency and re-check."""
        w = self.windows[key]
        op = self.ops[w.op_id]
        w.deps_remaining -= 1
        still = f"; still waiting on {w.deps_remaining}" if w.deps_remaining > 0 else ""
        self.engine.log("DecoderClstr",
                        f"decoder->decoder RECV: {op.name} W{w.k} <- {src_name} W{src_k}  "
                        f"(boundary arrived after t_dd){still}")
        self._check_window(key)
 
    def _deliver_to_orchestrator(self, op: Operation) -> None:
        # The decoder -> orchestrator hop (t_do, ~1 us) is paid for EVERY decoded window,
        # Clifford or not: the orchestrator always needs the result to update its frames.
        # Carry the operation's real decoded logical value (None if the decoder was a
        # timing-only stub, in which case the orchestrator uses its toy outcome).
        """Hand the op's decoded result to the orchestrator (the t_do hop) and stop. What happens
        next -- frame update, and DISPATCH of any conditional decision back to the chip
        (orchestrator -> controller -> chip) -- is the orchestrator's job, not the cluster's."""
        result = DecodeResult(op.id, self.nwin[op.id] - 1,
                              logical_value=self.op_results.get(op.id))
        self.orchestrator.integrate(op, result)