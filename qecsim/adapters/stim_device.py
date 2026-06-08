from __future__ import annotations
 
from ..message import Operation, SyndromePayload
# =============================================================
# STIM DEVICE ADAPTER
# =============================================================

class StimDevice:
    """Samples each operation's stim.Circuit and streams detection events per round."""
    def __init__(self):
        """Lazily import stim (raises if it is not installed)."""
        self._dets: dict = {}
        self._truth: dict = {}
        self._by_round: dict = {}
 
    def begin_operation(self, op: Operation) -> None:
        """Build and sample this operation's stim circuit."""
        import numpy as np  # noqa: F401
        sampler = op.circuit.compile_detector_sampler()
        dets, obs = sampler.sample(shots=1, separate_observables=True)
        self._dets[op.id] = dets[0]
        self._truth[op.id] = obs[0]
        coords = op.circuit.get_detector_coordinates()
        buckets: dict[int, list[int]] = {}
        for det_index, c in coords.items():
            t = int(c[-1])                 # last coordinate = round, set via SHIFT_COORDS
            buckets.setdefault(t, []).append(det_index)
        self._by_round[op.id] = buckets
 
    def round_payload(self, op: Operation, round_index: int) -> SyndromePayload:
        """Emit this round's REAL detection-event bits."""
        idx = self._by_round[op.id].get(round_index, [])
        bits = self._dets[op.id][idx]
        return SyndromePayload(op.id, op.patches[0], round_index, bits=bits,
                               coords=(0, 0, round_index))
