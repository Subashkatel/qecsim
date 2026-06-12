from __future__ import annotations

from typing import Callable, Optional

from ..message import Operation, SyndromePayload
# =============================================================
# STIM DEVICE ADAPTER
# =============================================================

class StimDevice:
    """Samples each operation's stim.Circuit and streams detection events per round.

    Round alignment (the convention the whole real-decoding path shares): chip round
    r (1-based) carries the detectors with stim time coordinate t = r - 1, and every
    layer PAST the chip's last round folds into the last round's payload -- a memory
    circuit with R noisy rounds has R+1 detector layers (t = 0..R; layer R is the
    final data-measurement layer), and the chip only asks for R rounds. The folded
    rule is round = min(t + 1, R). WindowErrorModels for the same op must be built
    with the same folded mapping (the cluster does this) so that a window's
    concatenated payload bits line up with its model's rows exactly.
    """
    def __init__(self, seed: Optional[int] = None,
                 rounds_for: Optional[Callable[[Operation], int]] = None):
        """`seed` makes the sample stream deterministic (one stateful sampler per op,
        re-sampled on every begin_operation, so repeated runs draw successive shots).
        `rounds_for` overrides the chip rounds R used for folding; default R = the
        circuit's highest detector time coordinate (the memory-experiment shape)."""
        self._seed = seed
        self._rounds_for = rounds_for
        self._samplers: dict = {}
        self._dets: dict = {}
        # the TRUE observable values of each sample -- not consumed by the timing
        # pipeline; retained for accuracy studies (compare against the decoder's
        # logical_value, e.g. the LogicalErrorRate metric stub in metrics.py).
        self._truth: dict = {}
        self._by_round: dict = {}

    def begin_operation(self, op: Operation) -> None:
        """Sample one fresh shot of this operation's stim circuit."""
        sampler = self._samplers.get(op.id)
        if sampler is None:
            sampler = op.circuit.compile_detector_sampler(seed=self._seed) \
                if self._seed is not None else op.circuit.compile_detector_sampler()
            self._samplers[op.id] = sampler
        dets, obs = sampler.sample(shots=1, separate_observables=True)
        self._dets[op.id] = dets[0]
        self._truth[op.id] = obs[0]
        coords = op.circuit.get_detector_coordinates()
        max_t = max((int(c[-1]) for c in coords.values()), default=0)
        R = self._rounds_for(op) if self._rounds_for is not None else max_t
        buckets: dict[int, list[int]] = {}
        for det_index, c in coords.items():
            t = int(c[-1])                 # last coordinate = round, set via SHIFT_COORDS
            buckets.setdefault(min(t + 1, R), []).append(det_index)
        for idx in buckets.values():
            idx.sort()                     # ascending detector id within each round
        self._by_round[op.id] = buckets

    def round_payload(self, op: Operation, round_index: int) -> SyndromePayload:
        """Emit this round's REAL detection-event bits."""
        idx = self._by_round[op.id].get(round_index, [])
        bits = self._dets[op.id][idx]
        patch = op.patches[0] if op.patches else (op.qubits[0] if op.qubits else 0)
        return SyndromePayload(op.id, patch, round_index, bits=bits)
