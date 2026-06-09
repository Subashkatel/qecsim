from __future__ import annotations
from .config import us
from .message import DecodeJob, DecodeResult
# ==================================================================================
# DECODERS
# The decoder implementations which is treated as a black box with two methods
# latency(job)-> ticks and Decode Results
# ==================================================================================

#TODO: Temperory stub decoder that models the latency of a real decoder without doing the work
class LatencyModelDecoder:
    """
    Latency model from Khalid et al., "Impacts of Decoder Latency on Utility-Scale Quantum
    Computer Architectures" (arXiv:2511.10633): single-unit decode time is the monomial
    tau_d(N) = alpha * N**beta (Eq. 12), where N is the node count of the decoding graph.
    Window decode time follows the paper's convention of per-round time times temporal extent
    -- e.g. a 3d-round window of a d^2-node patch costs tau_w = 3d * tau_d(d^2), and the memory
    reaction time is gamma_mem = 6d * tau_d(d^2) + t_com (Eq. 13) -- so latency(job) =
    n_rounds * tau_d(spatial_nodes).

    The default (alpha=2.85e-10 s, beta=1.2) is the paper's Table 3 fit for the Collision
    Cluster decoder on FPGA. Other Table 3 fits to swap in: Collision Cluster ASIC
    (5.53e-11, 1.34), AlphaQubit (4.8e-6, 0.503), PyMatching at p=0.1% (5.91e-9, 1.17).

    decode() is a stub (returns no logical value -- timing only), because in timing-only
    studies we don't need the actual correction.
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
    grounded in Maurya et al., "Real-time decoding of the gross code memory with FPGAs"
    (arXiv:2510.21600), with the FPGA-tailored decoder comparison in Maurya et al.,
    "FPGA-tailored algorithms for real-time decoding of quantum LDPC codes" (arXiv:2511.21660).

    Paper numbers (arXiv:2510.21600): one full Relay-BP iteration runs in 24 ns on FPGA
    (two 12 ns decoder cycles, Sec. 4.1/5); 12-round windows of the [[144,12,12]] gross code
    decode in < 240 ns on AVERAGE (~10 iterations, abstract); at p = 1e-3 the average is
    roughly 20 iterations (Sec. 7). The default iterations=40 is therefore a conservative
    WORST-CASE budget (a fixed iteration cap), not the paper's average -- pass iterations=10
    or 20 to reproduce the paper's average-case figures.
    """
    def __init__(self, iterations: int = 40, t_iter_ns: float = 24.0):
        """Store the BP iteration budget and the per-iteration FPGA time (ns)."""
        self.iterations = iterations
        self.t_iter_ns = t_iter_ns
 
    def latency(self, job: DecodeJob) -> int:
        """Decode time = iterations * time-per-iteration (per window), NOT a node monomial."""
        # ns -> us -> ticks. Independent of job.spatial_nodes by design: the FPGA evaluates
        # all check/variable nodes in parallel each cycle, so iteration time does not grow
        # with the graph (arXiv:2510.21600 Sec 4.1). Assumes windows sized as in the paper
        # (W = d = 12 rounds for the gross code); per-window cost is flat in n_rounds too.
        return us(self.iterations * self.t_iter_ns / 1000.0)
 
    def decode(self, job: DecodeJob) -> DecodeResult:
        """Return a result with NO logical value (TIMING-ONLY; a real Relay-BP fills it in)."""
        return DecodeResult(job.op_id, job.window_id, correction=None, logical_value=None,
                            latency_ticks=self.latency(job))
 
