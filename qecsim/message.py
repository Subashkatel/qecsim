from dataclasses import dataclass, field
import itertools
from typing import Callable, Optional, Any
#====================================================================================
# MESSAGE 
# Different types of messages that can be sent between the components in the system.
# Its designes as a dataclass because we might have new types of messages in the 
# future and we want to be able to easily add new fields to the messages without 
# breaking existing code.
#====================================================================================

@dataclass
class SyndromePayload:
    """This models one round of syndrome data for one logical operation.
    it flows from the chip to the controller to the decoder cluster to 
    the orchestrator and back to the controller and chip depending 
    on what kind of syndrome palyload it is."""
    operation_id: int
    patch_id: int
    round_index: int  # 1...R within this logical operation
    layer: str = "Z"  # "X" or "Z" layer (code-dependent))
    bits: Optional[Any] = None # GENERIC code-specific data; none for timingly-only mode
    coords: tuple = () # (x, y, t) space-time tag that keeps data seperated
    code: Optional[str] = None # code name for this syndrome (e.g. "surface", "color", "heavyhex", etc.)

dataclass
class MagicState:
    """ A distilled resource state a factory can hand to it consumers 
    (e.g. the orchestrator) to enable non-Clifford operations."""
    state_id: int
    fidelity: float = 1.0
    payload: Optional[Any] = None # GENERIC code-specific data; none for timingly-only mode

@dataclass
class DecodeJob:
    """ One unit of decode work : a window's rounds and its assembled payload alond with where to send the result when its done."""
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

@dataclass
class Window:
    """ One sliding window inside an operation's round stream."""
    op_id: int
    k: int # window index (0...W-1) within this logical operation
    commit_lo: int # the first round of this window commits 
    commit_hi: int # the last round of this window commits 
    buffer_hi: int # the last round of its look ahead buffer 
    n_rounds: int # number of rounds in this window 
    deps: list = field(default_factory=list) # window level dependencies (list of window indices that must be done before this one can start)
    committed: bool = False # has this window decode been commited 
    queued: bool = False # has this window beens placed on the ready queue for decoding
    blocked_logged: bool = False # set to true once we have logged that this window is blocked waiting for its dependencies so we don't spam the logs with the same message every round until the dependencies are done 

@dataclass
class WindowPlan:
    """ The full plan of how to break up one logical operation's round stream into windows and 
    what the dependencies between the windows are. it is compile time data that is sent to the 
    decoder cluster by the orchestrator before quantum executed, 0 simulated ticks"""
    windows: dict # the full set with deps/dependents and deps_remainig wired
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
    latency_ticks: int = 0 # how long the decode took in ticks (for logging and metrics)

@dataclass
class Decision:
    """ What the orchestrator returns to the chip (eg T gate measurment basis)"""
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
        """True iff this operation draws a distilled magic state form the factory (e.g. its a T gate or a non-Clifford gadget)."""
        if self.consumes_magic_state is not None:
            return self.consumes_magic_state
        return not self.clifford
    
