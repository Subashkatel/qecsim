from dataclasses import dataclass, field
from typing import Callable, Optional, Any
#====================================================================================
# MESSAGE 
# Different types of messages that can be sent between the components in the system.
# Each is designed as a dataclass because we might have new types of messages in the
# future and we want to be able to easily add new fields to the messages without
# breaking existing code.
#====================================================================================

@dataclass
class SyndromePayload:
    """This models one round of syndrome data for one logical operation.
    It flows from the chip to the controller to the decoder cluster to
    the orchestrator and back to the controller and chip depending
    on what kind of syndrome payload it is."""
    operation_id: int
    patch_id: int
    round_index: int  # 1...R within this logical operation
    bits: Optional[Any] = None # GENERIC code-specific data; None for timing-only mode
    code: Optional[str] = None # code name for this syndrome (e.g. "surface", "color", "heavyhex", etc.)
    n_fragments: int = 1 # how many payloads make up this op's round (one per PATCH when the device emits per-patch payloads). The CONTROLLER buffers a round's fragments and forwards them as one t_cd packet (arXiv:2511.10633 Sec III.1); the cluster's own fragment count is the receiving-side completeness backstop

    def __post_init__(self):
        if self.n_fragments < 1:
            raise ValueError(f"n_fragments must be >= 1 (got {self.n_fragments})")

@dataclass
class MagicState:
    """ A distilled resource state a factory can hand to its consumers
    (e.g. the orchestrator) to enable non-Clifford operations."""
    state_id: int
    fidelity: float = 1.0
    payload: Optional[Any] = None # GENERIC code-specific data; None for timing-only mode

@dataclass
class DecodeJob:
    """ One unit of decode work : a window's rounds and its assembled payloads, along with where to send the result when it's done."""
    op_id: int
    window_id: int
    n_rounds: int # commit + buffer rounds in this window
    dem: Optional[Any] = None  # detector error model for real decoders
    payloads: list = field(default_factory=list)
    ready_time: int = 0 # when it entered the ready queue (for logs)
    deadline: int = 0 # can be used by deadline-based schedulers to decide which job to do next
    on_done: Optional[Callable[[], None]] = None # set for externally scheduled jobs so they can trigger the next step in the pipeline when the decode is done
    label: str = "" # for logging purposes
    spatial_nodes: Optional[int] = None # Decoding graph nodes per round (spatial size of the decode job)
    code: Optional[str] = None  # which code this job belongs to (e.g. "surface", "color", "heavyhex", etc.) for code-specific decoding strategies or logs
    attempt: int = 0 # decode attempt number (0 = first pass); a DecoderRouter can escalate re-enqueued jobs (decoder switching, arXiv:2510.25222)
    hint: Optional[str] = None # routing hint for a DecoderRouter (e.g. "strong" for an escalated decoder-switching job); None = route normally
    window: Optional["Window"] = None # the Window this job decodes (None for external jobs). Gives the decoder the commit/buffer geometry it needs to place artificial defects (DecodeResult.boundary_defects) at the commit boundary -- the one place decoder and windowing scheme must agree (arXiv:2209.08552 Fig. 2)

@dataclass
class Window:
    """ One sliding window inside an operation's round stream."""
    op_id: int
    k: int # window index (0...W-1) within this logical operation
    commit_lo: int # the first round of this window commits
    commit_hi: int # the last round of this window commits
    buffer_hi: int # the last round of its look ahead buffer
    n_rounds: int # number of rounds in this window
    buffer_lo: Optional[int] = None # first round of a LEADING buffer (None = no leading buffer, i.e. commit_lo); windows of the parallel A/B scheme (arXiv:2511.10633 Sec II.4) have buffers on BOTH sides of the commit region
    deps: list = field(default_factory=list) # window level dependencies (list of window indices that must be done before this one can start)
    dependents: list = field(default_factory=list) # reverse edges: window keys that depend on THIS one (wired by the planner; the cluster sends each its boundary when this commits)
    deps_remaining: int = 0 # how many of `deps` are still uncommitted; the planner sets this to len(deps), the cluster counts it down as boundaries arrive
    committed: bool = False # has this window's decode been committed
    queued: bool = False # has this window been placed on the ready queue for decoding
    blocked_logged: bool = False # set to true once we have logged that this window is blocked waiting for its dependencies so we don't spam the logs with the same message every round until the dependencies are done
    boundary_in: dict = field(default_factory=dict) # artificial defects RECEIVED from committed predecessor windows, keyed {round | (round, patch_id): bit-mask} in THIS op's round numbering; XORed into the matching rounds' payloads when this window's job is assembled (never mutating the shared payload store)
    # ---- lifecycle timestamps (ticks; None until the stage is reached). Stamped by the
    # cluster at events it already handles -- zero new events, zero timing impact -- so a
    # WindowLatencyBreakdown metric can decompose every window's life into
    #   BUFFER-FILL (t_data_complete - t_first_round), DEP-BLOCK (t_queued - t_data_complete),
    #   QUEUE-WAIT (t_dispatch - t_queued), SERVICE (t_done - t_dispatch).
    t_first_round: Optional[int] = None   # this window's first round arrived
    t_data_complete: Optional[int] = None # all commit+buffer rounds present
    t_queued: Optional[int] = None        # data AND dependencies met -> ready queue
    t_dispatch: Optional[int] = None      # popped by a free decoder unit
    t_done: Optional[int] = None          # decode finished, window committed

    @property
    def start_round(self) -> int:
        """First round this window needs (leading buffer if present, else commit start)."""
        return self.commit_lo if self.buffer_lo is None else self.buffer_lo

@dataclass
class WindowPlan:
    """ The full plan of how to break up each logical operation's round stream into windows and
    what the dependencies between the windows are. It is compile-time data, sent to the
    decoder cluster by the orchestrator before quantum execution begins (0 simulated ticks)."""
    windows: dict # the full set with deps/dependents and deps_remaining wired
    nwin: dict # number of windows
    op_windows: dict # list of window indices per op
    successors: dict # the forward edges of the operation DAG (this is operation level direction where as the deps is window level direction)
    spatial_nodes: dict # spatial size of each operation's decoding job
    total_windows: int # total number of windows across all operations (for logging and metrics)
    summary: dict = field(default_factory=dict) # any extra code-specific data about the window plan that we want to log or use in the decoder (e.g. for a surface code we might want to include the code distance and for a color code we might want to include the number of qubits)

@dataclass
class DecodeResult:
    """ What a decoder returns for one window when its done decoding : the logical value and the correction"""
    op_id: int
    window_id: int
    correction: Optional[Any] = None # the pauli correction the decoder produces
    logical_value: Optional[int] = None # the decode logical outcome
    soft_output: Optional[float] = None # decoder confidence in its estimate (arXiv:2510.25222 Sec II.2: complementary/cluster gap); None for decoders without soft output. A low value is what triggers a switch to a stronger decoder.
    # ARTIFICIAL DEFECTS (arXiv:2209.08552 Sec I.B/Fig. 2; arXiv:2511.10633 Sec II.4):
    # committing only the in-commit-region part of correction chains that cross the commit
    # boundary "will introduce new defects, referred to as 'artificial defects' along the
    # boundary" -- nodes just OUTSIDE the commit region. The dependent window must include
    # them in its own defect data. Convention: {round_index: bit-mask} in the SOURCE op's
    # round numbering (cross-op dependents are shifted by the cluster), where the mask is
    # XORed into that round's syndrome bits at job-assembly time. Use (round_index,
    # patch_id) keys for per-patch precision when rounds have multiple payload fragments.
    # None = decoder produces no boundary data (timing-only decoders).
    boundary_defects: Optional[dict] = None

@dataclass
class Decision:
    """ What the orchestrator returns to the chip (e.g. a T gate measurement basis)"""
    gadget_id: int # the operation id of the gadget this instruction is for
    basis: str # e.g. "X" or "Z" for a T gate measurement basis decision; could be more general for other types of decisions in the future

@dataclass
class Operation:
    """ One logical operation in the circuit (its patches, predecessors, and any gating information)."""
    id: int
    name: str
    qubits: tuple
    clifford: bool = True
    gated_by: Optional[int] = None # non clifford: previous operation that gates this one 
    circuit: Optional[Any] = None  # a stim.Circuit 
    consumes_magic_state: Optional[bool] = None # does this operation consume a magic state 
    patches: tuple = () # the patches involved in this operations (for routing and window planning)
    predecessors: tuple = () # operation ids that share a patch and run before this one (for routing and window planning)
    has_successor: bool = False # does this operation have a successor that shares a patch and runs after it 

    @property
    def needs_magic_state(self) -> bool :
        """True iff this operation draws a distilled magic state from the factory (e.g. it's a T gate or a non-Clifford gadget)."""
        if self.consumes_magic_state is not None:
            return self.consumes_magic_state
        return not self.clifford
    
