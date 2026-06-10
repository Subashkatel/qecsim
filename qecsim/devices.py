from __future__ import annotations
 
from typing import TYPE_CHECKING
 
from .message import Operation, SyndromePayload
 
if TYPE_CHECKING:                      
    from .protocols import CodeModel

#===============================================================================
# DEVICES
# This module implements the QPU-side syndrome source; the chip calls it each
# round to get that round's payload(s).
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
                               round_index)
    
#TODO: This emits fake syndrome bits upate it to use stim to get real detection events
class SyndromeBitDevice:
    """Provides syndrome-bit payloads per round so a real decoder has data to work on.
    THIS IS CURRENTLY USING FAKE (PSEUDO-RANDOM) SYNDROME BITS the values are not from a
    real quantum simulation; they exist only to exercise the full payload path (cluster
    buffer -> window assembly -> decoder). Swap in StimDevice for true detection events; the
    rest of the pipeline is unchanged.

    With per_patch=True it emits ONE payload PER PATCH each round (the round_payloads
    hook), each sized to a single patch -- the granularity that patch-local decoding
    graphs and the spatial windows of lattice surgery eventually need (arXiv:2511.10633
    gamma_LS). The default (per_patch=False) keeps the original one aggregated payload."""
    def __init__(self, code: CodeModel, seed: int = 0, max_bits: int = 8,
                 per_patch: bool = False):
        """Seed a small deterministic bit generator sized to the code."""
        import random
        self.code = code
        self.max_bits = max_bits
        self.per_patch = per_patch
        self.rng = random.Random(seed)

    def begin_operation(self, op: Operation) -> None:
        """Nothing to set up."""
        pass

    def _bits(self, num_patches: int) -> list:
        """Fake bits for one payload covering this many patches."""
        nbits = min(self.code.syndrome_bits_per_round(num_patches), self.max_bits)
        return [self.rng.randint(0, 1) for _ in range(nbits)]

    def round_payload(self, op: Operation, round_index: int) -> SyndromePayload:
        """Emit this round's syndrome bits. NOTE: THESE ARE FAKE (PSEUDO-RANDOM) BITS."""
        return SyndromePayload(op.id, op.patches[0] if op.patches else op.qubits[0],
                               round_index, bits=self._bits(len(op.qubits)),
                               code=self.code.name)

    def round_payloads(self, op: Operation, round_index: int) -> list[SyndromePayload]:
        """One payload per patch when per_patch=True; else the single aggregated payload."""
        if not self.per_patch:
            return [self.round_payload(op, round_index)]
        patches = op.patches if op.patches else op.qubits
        return [SyndromePayload(op.id, p, round_index, bits=self._bits(1),
                                code=self.code.name)
                for p in patches]
