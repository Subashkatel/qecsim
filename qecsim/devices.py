from __future__ import annotations
 
from typing import TYPE_CHECKING
 
from .message import Operation, SyndromePayload
 
if TYPE_CHECKING:                      
    from .protocols import CodeModel

#===============================================================================
# DEVICES
# This module implements the qpu side syndrom source to the chip calls each round
# to get that payload 
#===============================================================================

#TODO: This is just simple timing only it doesn't emit actual syndrom bits.
class TimingOnlyDevice:
    """Emits payloads with no bits. Fast: studies timing only."""
    def begin_operation(self, op: Operation) -> None:
        """Nothing to set up for this device."""
        pass
 
    def round_payload(self, op: Operation, round_index: int) -> SyndromePayload:
        """Emit an EMPTY payload (timing-only; carries no syndrome bits)."""
        return SyndromePayload(op.id, op.patches[0] if op.patches else op.qubits[0],
                               round_index, coords=(0, 0, round_index))
    
#TODO: This emits fake syndrome bits upate it to use stim to get real detection events
class SyndromeBitDevice:
    """Provides one syndrome-bit payload per round so a real decoder has data to work on.
    THIS IS CURRENTLY USING FAKE (PSEUDO-RANDOM) SYNDROME BITS the values are not from a
    real quantum simulation; they exist only to exercise the full payload path (cluster
    buffer -> window assembly -> decoder). Swap in StimDevice for true detection events; the
    rest of the pipeline is unchanged."""
    def __init__(self, code: CodeModel, seed: int = 0, max_bits: int = 8):
        """Seed a small deterministic bit generator sized to the code."""
        import random
        self.code = code
        self.max_bits = max_bits
        self.rng = random.Random(seed)
 
    def begin_operation(self, op: Operation) -> None:
        """Nothing to set up."""
        pass
 
    def round_payload(self, op: Operation, round_index: int) -> SyndromePayload:
        """Emit this round's syndrome bits. NOTE: THESE ARE FAKE (PSEUDO-RANDOM) BITS."""
        nbits = min(self.code.syndrome_bits_per_round(len(op.qubits)), self.max_bits)
        bits = [self.rng.randint(0, 1) for _ in range(nbits)]
        return SyndromePayload(op.id, op.patches[0] if op.patches else op.qubits[0],
                               round_index, bits=bits, coords=(0, 0, round_index),
                               code=self.code.name)
