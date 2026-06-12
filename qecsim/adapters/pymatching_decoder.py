from __future__ import annotations

from typing import TYPE_CHECKING

from ..message import DecodeJob, DecodeResult

if TYPE_CHECKING:
    from ..protocols import Decoder

# =====================================================================================
# PYMATCHING DECODER ADAPTER
# =====================================================================================

class PyMatchingDecoder:
    """Real decoding for correctness; latency still comes from a latency model.

    The job's decoding problem is its window's WindowErrorModel (job.dem, filled by
    the cluster at plan load): the window's slice of the operation's global detector
    error model. Per window we cache one pymatching.Matching built from the model's
    check matrix and priors; the syndrome is the job's assembled payload bits (the
    cluster already XORed received artificial defects in). Of the matching's selected
    faults we keep only the OWNED columns (this window's commit responsibility):
    their observable flips become DecodeResult.logical_value, and their detector
    flips beyond the commit boundary become DecodeResult.boundary_defects -- per-round
    bit masks (the convention Window.boundary_in speaks), which the cluster ships to
    the dependent windows (arXiv:2209.08552 Sec I.B's artificial defects).

    Scope: single-patch operations and a single logical observable per op (the
    cluster XOR-accumulates logical_value as one int per op). Timing-only jobs
    (dem=None, e.g. ops without circuits or A/B-scheme windows) return an empty
    result, exactly like a timing-only stub."""
    def __init__(self, latency_model: Decoder):
        """Reuse a latency model for timing; pymatching imports lazily on first decode."""
        self.latency_model = latency_model
        self._matchings: dict = {}     # id(model) -> cached pymatching.Matching

    def latency(self, job: DecodeJob) -> int:
        """Timing comes from the wrapped latency model."""
        return self.latency_model.latency(job)   # TIME from the model, not wall-clock

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Run real minimum-weight matching on the job's window error model."""
        model = job.dem
        if model is None:                        # timing-only job: no real data to decode
            return DecodeResult(job.op_id, job.window_id)
        import numpy as np
        import pymatching
        m = self._matchings.get(id(model))
        if m is None:
            weights = np.log((1 - model.priors) / model.priors)
            m = pymatching.Matching.from_check_matrix(model.check, weights=weights)
            self._matchings[id(model)] = m
        syndrome = np.concatenate(
            [np.asarray(p.bits, dtype=np.uint8) for p in job.payloads
             if p.bits is not None]) if job.payloads else np.zeros(0, dtype=np.uint8)
        if syndrome.size != model.check.shape[0]:
            raise ValueError(
                f"{job.label}: payload bits ({syndrome.size}) do not match the window "
                f"error model's detectors ({model.check.shape[0]}) -- the device and "
                "the cluster's model build must share the folded round convention")
        selected = np.asarray(m.decode(syndrome), dtype=np.uint8)
        committed = selected.astype(bool) & model.owned
        obs_flips = (model.obs @ committed.astype(np.uint8)) % 2
        defects: dict = {}
        for col in np.nonzero(committed)[0]:
            for det in model.future_flips.get(int(col), ()):
                r, pos = model.defect_positions[det]
                mask = defects.setdefault(r, [])
                if len(mask) <= pos:
                    mask.extend([0] * (pos + 1 - len(mask)))
                mask[pos] ^= 1
        return DecodeResult(job.op_id, job.window_id,
                            correction=committed.astype(np.uint8),
                            logical_value=int(obs_flips[0]) if obs_flips.size else 0,
                            boundary_defects=defects or None)
