from __future__ import annotations
from typing import Any, Callable, Optional, Protocol, runtime_checkable
from qecsim.message import (Operation, SyndromePayload, DecodeJob, Window, WindowPlan,
                      DecodeResult, Decision)
#=========================================================================
# PROTOCOLS 
# This module defines the abstract base classes for various differnt
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
    """ This will help us with emits syndromes rounds. The default emits
    empty (timing only) payloads but round_payload returns one round's of 
    SyndromePayloads with code-specific data for real decoders."""
    def begin_operation(self, op:Operation) -> None:...
    def round_payload(self, op:Operation, round_index:int) -> SyndromePayload:...

@runtime_checkable
class CodeModel(Protocol):
    """ This will provide all the code-specific quantity the control/decode side
    needs to know to do its job."""
    @property
    def name(self) -> str:...
    @property
    def distance(self) -> int:...
    def rounds_per_logical_cycle(self) -> int:...
    def commit_rounds(self) -> int:...
    def buffer_rounds(self) -> int:...
    def spatial_nodes(self, num_patches: int) -> int:...
    def syndrome_bits_per_round(self) -> int:...

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
    handed off. It organizes the syndrome stream into decodable chunks."""
    def plan_windows(self, op_id: int, n_rounds: int,
                     code: CodeModel) -> list[tuple[int, int, int]]: ...
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
class Decoder(Protocol):
    """ This receives a DecodeJob and answer two seperate questions:
    latency(job) = how many ticks the decode takes 
    (this is what advances the simulated clock), and decode(job) = 
    the actual logical result/correction."""

    def latency(self, job: DecodeJob) -> int: ...
    def decode(self, job: DecodeJob) -> DecodeResult: ...

@runtime_checkable
class DecoderService(Protocol):
    """Lets any component submit a decode job to the decoder cluster and get back the result when its done
    This is how the factorys correction-qubit decoding is routed through the REAL cluster instead of being
    abstracted away."""
    def submit_decode(self, n_rounds: int, on_done: Callable[[], None],
                      label: str = ..., deadline: Optional[int] = ...) -> None: ...


@runtime_checkable
class Controller(Protocol):
    """ The controller is responsible for taking the syndrome stream from the chip and 
    routing it to the decoder cluster, and then taking the decode results and 
    routing them back to the chip to apply the corrections."""
    def relay_syndrome(self, payload: SyndromePayload,
                       deliver: Callable[[SyndromePayload], None]) -> None: ...
    def relay_instruction(self, decision: Decision,
                          deliver: Callable[[Decision], None]) -> None: ...
    def dec_to_dec_delay(self) -> int: ...
    def dec_to_orch_delay(self) -> int: ...

@runtime_checkable
class Orchestrator(Protocol):
    """ """
    

@runtime_checkable
class MagicStateFactory(Protocol):
    """ """
    def request(self, op_id: int, callback: Callable[[], None]) -> None: ...
 
 
@runtime_checkable
class QuantumProcessor(Protocol):
    """ """
    def load(self, ops: list) -> None: ...
    def on_decision(self, decision: Decision) -> None: ...
 
 
@runtime_checkable
class Metric(Protocol):
    """ """
    def observe(self, engine: "Engine") -> None: ...
    def result(self): ...