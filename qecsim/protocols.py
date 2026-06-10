from __future__ import annotations
from typing import Any, Callable, Optional, Protocol, runtime_checkable, TYPE_CHECKING
from .message import (Operation, SyndromePayload, DecodeJob, Window, WindowPlan,
                      DecodeResult, Decision)

if TYPE_CHECKING:                      # type-only; Metric.observe references the Engine
    from .engine import Engine
#=========================================================================
# PROTOCOLS 
# This module defines the abstract base classes for various different
# things making the whole system modular and swappable. For example,
# we can have different decoding strategies that can be swapped in 
# and out without changing the rest of the system. Or we can have 
# different types of chips that can be swapped in and out without 
# changing the rest of the system.
#=========================================================================

@runtime_checkable
class InputFrontend(Protocol):
    """ The input frontend is responsible for taking in the raw input and 
    converting it into a operation DAG the engine runs"""
    def build(self) -> list[Operation]:...

@runtime_checkable
class DeviceModel(Protocol):
    """ The QPU-side syndrome source: round_payload returns one round's SyndromePayload.
    The default device emits empty (timing-only) payloads; real devices fill in
    code-specific detection bits for real decoders.

    OPTIONAL extension hook (duck-typed; the chip falls back to round_payload):
        round_payloads(op, round_index) -> list[SyndromePayload]
    emits one payload PER PATCH for the round (each tagged with n_fragments by the chip)
    -- the granularity patch-local decoding graphs and spatial lattice-surgery windows
    need (arXiv:2511.10633 gamma_LS). The cluster counts a round as arrived only once all
    its fragments are in."""
    def begin_operation(self, op:Operation) -> None:...
    def round_payload(self, op:Operation, round_index:int) -> SyndromePayload:...

@runtime_checkable
class CodeModel(Protocol):
    """ This will provide all the code-specific quantity the control/decode side
    needs to know to do its job. like distance, rounds per logical cycle, commit/buffer
    rounds, decode-graph node and syndrome bit counts."""
    @property
    def name(self) -> str:...
    @property
    def distance(self) -> int:...
    def rounds_per_logical_cycle(self) -> int:...
    def commit_rounds(self) -> int:...
    def buffer_rounds(self) -> int:...
    def spatial_nodes(self, num_patches: int) -> int:...
    def syndrome_bits_per_round(self, num_patches: int) -> int:...

@runtime_checkable
class LayoutModel(Protocol):
    """ The QPU LAYOUT : which code each PATCH is running ..."""
    @property
    def name(self) -> str: ...
    @property
    def distance(self) -> int: ...
    def code_for_patch(self, patch_id: Any) -> CodeModel: ... # which code is this tile encoded in?"
    def code_for_op(self, op: Operation) -> CodeModel: ...  # which codes is this operation running on (used to set the window's commit/buffer timing)
    def spatial_nodes_for(self, op: Operation) -> int: ...
    def codes(self) -> list: ...

@runtime_checkable
class DecodingScheme(Protocol):
    """plan_windows decides the window boundaries (commit/buffer rounds),
    and data_complete decides when a window has enough rounds to be
    handed off. It organizes the syndrome stream into decodable chunks.

    plan_windows returns 3-tuples (commit_lo, commit_hi, buffer_hi) or 4-tuples
    (buffer_lo, commit_lo, commit_hi, buffer_hi) for windows with a LEADING buffer.

    OPTIONAL extension hooks (duck-typed; the WindowPlanner falls back to the sequential
    chain when absent, so existing schemes need not implement them). The dependency
    STRUCTURE between windows is part of the windowing strategy -- a sequential scheme
    chains its windows, the parallel A/B scheme (arXiv:2511.10633 Sec II.4) makes layer-B
    windows depend on their two layer-A neighbours:
        wire_deps(windows: list[Window]) -> None       # set deps among ONE op's windows
        entry_windows(windows) -> list[Window]         # receive predecessor-op boundaries
        exit_windows(windows) -> list[Window]          # emit boundaries to successor ops
    """
    def plan_windows(self, op_id: int, n_rounds: int,
                     code: CodeModel) -> list[tuple]: ...
    def data_complete(self, window: Window, rounds_arrived: int, successor_rounds: int,
                      memory_rounds: int, n_rounds: int, has_successor: bool,
                      op: Operation = None, layout: LayoutModel = None) -> bool: ...


@runtime_checkable
class ExecutionPlanner(Protocol):
    """ This is responsible for taking in the operation DAG and coming up with a 
    full plan of how to break up each operation's round stream into windows and 
    when to schedule each window for decoding based on its dependencies."""
    def plan(self, ops: list[Operation]) -> WindowPlan: ...

@runtime_checkable
class RoundsPolicy(Protocol):
    """ This decides how many QEC rounds an operation runs for. It is swappable so we can
    test a fixed global round count (FixedRounds), per-code round counts (CodeRounds), or
    counts computed from anything else about the operation and its code."""
    def rounds_for(self, op: "Operation", code: "CodeModel") -> int: ...

@runtime_checkable
class Decoder(Protocol):
    """ This receives a DecodeJob and answers two separate questions:
    latency(job) = how many ticks the decode takes
    (this is what advances the simulated clock), and decode(job) =
    the actual logical result/correction.

    BOUNDARY-DATA convention (windowed real decoding; arXiv:2209.08552 Sec I.B/Fig. 2,
    arXiv:2511.10633 Sec II.4): a windowed decoder commits only the correction edges
    inside the commit region (job.window has the geometry); for chains that CROSS out of
    it, it reports artificial defects -- the nodes just outside the commit region on such
    edges -- as DecodeResult.boundary_defects = {round (or (round, patch_id)): bit-mask}
    in the op's round numbering. The cluster ships them over the t_dd hop and XORs them
    into the dependent window's payloads at assembly. Timing-only decoders return None
    (the hop still clears the dependency). NOTE (arXiv:2209.08552): the inner decoder
    must return approximately LOW-WEIGHT corrections (MWPM/UF) -- homology-class decoders
    create spurious artificial defects."""

    def latency(self, job: DecodeJob) -> int: ...
    def decode(self, job: DecodeJob) -> DecodeResult: ...

@runtime_checkable
class Scheduler(Protocol):
    """ This is the decode cluster's queue-ordering policy for the ready queue of decode
    jobs. insert puts a job in the queue, and pop chooses the next job to run."""
    def insert(self, queue: list, job: DecodeJob) -> None: ...
    def pop(self, queue: list) -> DecodeJob: ...


@runtime_checkable
class DeadlinePolicy(Protocol):
    """Assigns each window job its deadline when it is enqueued -- what gives a
    deadline-aware Scheduler (EDF) something real to order by. `on_reaction_path` is True
    when the op's decode result gates a waiting non-Clifford gate (the reaction path of
    arXiv:2511.10633); the default EnqueueTimeDeadline ignores it (deadline = now, so EDF
    behaves like FIFO), ReactionPathDeadline prioritizes it."""
    def deadline(self, op: Operation, window: Window, now: int,
                 on_reaction_path: bool) -> int: ...


@runtime_checkable
class DecoderRouter(Protocol):
    """Picks the decoder for each DecodeJob at dispatch time. The default CodeRouter
    routes by job.code (per-code decoders, G1); a custom router can route by job.hint /
    job.attempt -- the seam needed for decoder switching (escalate a low-confidence window
    to a strong decoder, arXiv:2510.25222) and for clusters mixing decoding devices with
    different latency models (FPGA / ASIC / GPU / CPU)."""
    def route(self, job: DecodeJob) -> "Decoder": ...

@runtime_checkable
class DecoderService(Protocol):
    """Lets any component submit a decode job to the decoder cluster and get back the result
    when it's done. This is how the factory's correction-qubit decoding is routed through the
    REAL cluster instead of being abstracted away."""
    def submit_decode(self, n_rounds: int, on_done: Callable[[], None],
                      label: str = ..., deadline: Optional[int] = ...,
                      code: Optional[str] = ...,
                      spatial_nodes: Optional[int] = ...) -> None: ...


@runtime_checkable
class WorkloadManager(DecoderService, Protocol):
    """The decoder-cluster seam: the paper's "workload manager" (arXiv:2511.10633 Sec III)
    as the chip and the wiring see it. This is the EXPLICIT contract a custom cluster must
    satisfy:
      - register_op / build_windows / load_execution_plan: install the workload,
      - rounds_for: the agreed temporal length of each operation (chip cadence reads it),
      - on_syndrome_arrival / on_memory_round: the controller's delivery targets,
      - submit_decode (inherited DecoderService): external jobs (factory, idle rounds).
    Documented ATTRIBUTES the standard wiring also relies on (not isinstance-checked):
      layout, scheme, rounds_policy   -- read to build the default WindowPlanner; a custom
                                         manager without them must be paired with planner=
      on_workload_complete            -- lifecycle sink the wiring sets (factory shutdown)
      queue_log                       -- optional [(t, queue_len)] trace for the summary"""
    def register_op(self, op: Operation) -> None: ...
    def build_windows(self) -> None: ...
    def load_execution_plan(self, plan: WindowPlan) -> None: ...
    def rounds_for(self, op: Operation) -> int: ...
    def on_syndrome_arrival(self, payload: SyndromePayload) -> None: ...
    def on_memory_round(self, op_id: int) -> None: ...


@runtime_checkable
class Controller(Protocol):
    """ The controller is responsible for taking the syndrome stream from the chip and 
    routing it to the decoder cluster, and then taking the decode results and 
    routing them back to the chip to apply the corrections."""
    def relay_syndrome(self, payload: SyndromePayload,
                       deliver: Callable[[SyndromePayload], None]) -> None: ...
    def relay_instruction(self, decision: Decision,
                          deliver: Callable[[Decision], None]) -> None: ...

@runtime_checkable
class Orchestrator(Protocol):
    """The orchestrator seam (arXiv:2511.10633 Sec III): owns the Pauli frames, integrates
    decode results into them, decides conditional logical operations from decoded outcomes,
    and dispatches those decisions back to the chip through the controller. announce_plan
    marks the compile-time handoff of the window/job plan to the decoder cluster (0 ticks,
    off the reaction path). Swap this to study a different orchestration policy."""
    def connect(self, controller: "Controller", decision_sink: Callable) -> None: ...
    def register_gate(self, gated_op_id: int, gating_op_id: int) -> None: ...
    def announce_plan(self, plan: WindowPlan) -> None: ...
    def integrate(self, op: Operation, result: DecodeResult) -> None: ...
    def on_result(self, op: Operation, result: DecodeResult) -> list[Decision]: ...
    

@runtime_checkable
class MagicStateFactory(Protocol):
    """A non-Clifford gate asks the factory for a distilled magic state; this seam decides
    whether the state is in stock or the gate has to wait (the supply stall)."""
    def request(self, op_id: int, callback: Callable[[], None]) -> None: ...
    def shutdown(self) -> None: ...   # stop background production; wired to cluster.on_workload_complete

@runtime_checkable
class QuantumProcessor(Protocol):
    """The QPU seam: drives the round cadence, emits syndromes through the controller,
    enforces data dependencies and T-gate gating, and receives the decoded correction back.
    load(ops) begins executing the DAG; on_decision(decision) receives a returned
    correction (releasing a blocked gate). `last_finish_time` is a documented attribute,
    not a checked protocol member."""
    def load(self, ops: list) -> None: ...
    def on_decision(self, decision: Decision) -> None: ...
 
 
@runtime_checkable
class Metric(Protocol):
    """A pluggable observer. The engine calls observe() after every event (a no-op when no
    metrics are registered, so adding metrics never changes the timing or the trace), and
    result() returns the final value."""
    def observe(self, engine: "Engine") -> None: ...
    def result(self): ...