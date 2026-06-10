from __future__ import annotations
 
from typing import TYPE_CHECKING
 
from ..message import DecodeJob, DecodeResult

if TYPE_CHECKING:
    from ..protocols import Decoder
 
# =====================================================================================
# PYMATCHING DECODER ADAPTER
# =====================================================================================

# TODO: currently just a stub -- skeleton adapter; needs the `pymatching` package for real decoding.
class PyMatchingDecoder:
    """Real decoding for correctness; latency still comes from a latency model.

    TODO (boundary defects, the windowed-decoding contract -- see Decoder protocol):
    to participate in real sliding-window decoding this adapter should also fill
    DecodeResult.boundary_defects. Recipe (arXiv:2209.08552 Sec I.B): use
    `matching.decode_to_edges_array(syndrome)` to get the matched edges as detector-index
    pairs; map detector indices to (round, position) via the DEM's detector coordinates
    (StimDevice already buckets detectors by their last coordinate = round); for every
    matched edge with one endpoint inside the commit region (job.window.commit_lo ..
    commit_hi) and the other outside it, flip the OUTSIDE endpoint's bit in the mask for
    its round: boundary_defects[round][bit_index] ^= 1. Blocked on per-window DEMs
    (job.dem is not yet populated by the standard wiring -- docs/README.md gap #7)."""
    def __init__(self, latency_model: Decoder):
        """Lazily import pymatching; reuse a latency model for timing."""
        self.latency_model = latency_model
 
    def latency(self, job: DecodeJob) -> int:
        """Timing comes from the wrapped latency model."""
        return self.latency_model.latency(job)   # TIME from the model, not wall-clock
 
    def decode(self, job: DecodeJob) -> DecodeResult:
        """Run real minimum-weight-matching decoding on the job's error model."""
        import pymatching
        m = pymatching.Matching.from_detector_error_model(job.dem)
        import numpy as np
        syndrome = np.concatenate([p.bits for p in job.payloads if p.bits is not None])
        prediction = m.decode(syndrome)
        return DecodeResult(job.op_id, job.window_id, correction=prediction,
                            logical_value=int(prediction.sum() % 2))