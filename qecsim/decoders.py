from __future__ import annotations
from config import us
from message import DecodeJob, DecodeResult
# ==================================================================================
# DECODERS
# The decoder implementations which is treated as a black box with two methods
# latency(job)-> ticks and Decode Results
# ==================================================================================

#TODO: Temperory stub decoder that models the latency of a real decoder without doing the work
class LatencyModelDecoder:
    """
    Latency from the paper's  tau_d(N) = alpha * N**beta  per round,
    times the number of rounds in the window. decode() is a stub (returns no logical value --
    timing only), because in timing-only studies we don't need the actual correction.
    Defaults are a stand-in for a fast collision-cluster-style decoder.
    """
    def __init__(self, d: int, alpha: float = 2.85e-10, beta: float = 1.2):
        """Store the latency coefficients (alpha, beta) and the code distance."""
        self.d = d
        self.alpha = alpha
        self.beta = beta
 
    def latency(self, job: DecodeJob) -> int:
        # spatial size of the decoding graph: use the job's value (set per window from the
        # operation's patch count) if given, else fall back to a single d-by-d patch.
        """Decode time = alpha * nodes^beta per round, times the window's rounds (ticks)."""
        n_nodes = job.spatial_nodes if job.spatial_nodes else self.d * self.d
        per_round = self.alpha * (n_nodes ** self.beta)
        return us(job.n_rounds * per_round * 1e6)
 
    def decode(self, job: DecodeJob) -> DecodeResult:
        """Return a result with NO logical value (TIMING-ONLY; a real decoder fills it in)."""
        return DecodeResult(job.op_id, job.window_id, correction=None, logical_value=None,
                            latency_ticks=self.latency(job))

#TODO: Fix these these are temporory stubs for testing 
class PresetLatencyDecoder:
    """A decoder whose decode time is A SINGLE CONSTANT you provide (it does NOT depend on
    window size), handy for hand-checking timing (e.g. 'every decode takes 1 microsecond').
    decode() is a stub (returns no logical value). Swap in wherever a Decoder is expected."""
    def __init__(self, latency_us: float = 1.0):
        """Store the single constant decode time to use for every job."""
        self._lat = us(latency_us)
 
    def latency(self, job: DecodeJob) -> int:
        """Always return the same decode time (independent of window size)."""
        return self._lat
 
    def decode(self, job: DecodeJob) -> DecodeResult:
        """Return a result with NO logical value (TIMING-ONLY)."""
        return DecodeResult(job.op_id, job.window_id, latency_ticks=self._lat)


#TODO: Fix these these are temporory stubs for testing
class ParityDecoder:
    """A minimal REAL decoder: it actually consumes the window's assembled payloads and
    returns a logical value (here, the parity of all payload bits). Latency reuses the
    alpha*N**beta model. This exists to exercise the full data path end to end -- payload in,
    logical value out, which the orchestrator turns into the T-gate correction. A production
    decoder (e.g. PyMatchingDecoder) implements the same two methods over a real DEM."""
    def __init__(self, d: int = 3, alpha: float = 2.85e-10, beta: float = 1.2):
        """Reuse the latency model for timing; track how many payloads were seen."""
        self._lat = LatencyModelDecoder(d=d, alpha=alpha, beta=beta)
        self.payloads_seen = 0          # diagnostic: how many payloads reached the decoder
 
    def latency(self, job: DecodeJob) -> int:
        """Same decode time as the latency-model decoder."""
        return self._lat.latency(job)
 
    def decode(self, job: DecodeJob) -> DecodeResult:
        """Return the parity of all payload bits. NOTE: TOY DECODER, not real QEC decoding."""
        bits: list = []
        for p in job.payloads:
            if p.bits:
                bits += list(p.bits)
        self.payloads_seen += len(job.payloads)
        return DecodeResult(job.op_id, job.window_id, correction=None,
                            logical_value=(sum(bits) % 2),
                            latency_ticks=self._lat.latency(job))

#TODO: Fix these these are temporory stubs for testing
class RelayBPDecoder:
    """A latency model for BELIEF-PROPAGATION decoding of QLDPC / bivariate-bicycle codes,
    grounded in Maurer et al., "Real-time decoding of the gross code memory with FPGAs"
    (arXiv:2510.21600), with the FPGA-tailored comparison in arXiv:2511.21660.
    """
    def __init__(self, iterations: int = 40, t_iter_ns: float = 24.0):
        """Store the BP iteration budget and the per-iteration FPGA time (ns)."""
        self.iterations = iterations
        self.t_iter_ns = t_iter_ns
 
    def latency(self, job: DecodeJob) -> int:
        """Decode time = iterations * time-per-iteration (per window), NOT a node monomial."""
        # ns -> us -> ticks. Independent of job.spatial_nodes by design
        return us(self.iterations * self.t_iter_ns / 1000.0)
 
    def decode(self, job: DecodeJob) -> DecodeResult:
        """Return a result with NO logical value (TIMING-ONLY; a real Relay-BP fills it in)."""
        return DecodeResult(job.op_id, job.window_id, correction=None, logical_value=None,
                            latency_ticks=self.latency(job))
 