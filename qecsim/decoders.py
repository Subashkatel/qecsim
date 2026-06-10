from __future__ import annotations
from typing import TYPE_CHECKING
from .config import us
from .message import DecodeJob, DecodeResult

if TYPE_CHECKING:
    from .protocols import Decoder
# ==================================================================================
# DECODERS
# The decoder implementations. A decoder is a black box with two methods:
# latency(job) -> ticks and decode(job) -> DecodeResult.
# ==================================================================================

class CodeRouter:
    """The default DecoderRouter: route each job by its CODE (G1). Falls back to the
    default decoder when the job's code has no dedicated entry -- so a single-code run
    always uses the one default decoder, byte-identical to before routing existed.
    A custom router can route by job.hint / job.attempt instead (e.g. escalate a
    low-confidence window to a strong decoder, arXiv:2510.25222), or model a cluster
    mixing decoding devices with different speeds (FPGA / ASIC / GPU latency models)."""
    def __init__(self, default, by_code: dict = None):
        """Hold the default decoder and the optional {code_name: Decoder} map."""
        self.default = default
        self.by_code = dict(by_code) if by_code else {}

    def route(self, job: DecodeJob):
        """Pick the decoder for this job (by code; default when unmapped)."""
        return self.by_code.get(job.code, self.default)

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
        return DecodeResult(job.op_id, job.window_id, correction=None, logical_value=None)

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
        return DecodeResult(job.op_id, job.window_id)


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
                            logical_value=(sum(bits) % 2))

class SwitchingDecoder:
    """TIMING-LEVEL decoder switching (Toshio et al., arXiv:2510.25222): a fast weak
    decoder backed by a slow strong decoder. With probability `gamma_switch` a window's
    weak decode is unreliable (soft output below threshold, Sec III.1) and the job pays
        tau_weak + 2*handoff + tau_strong
    (weak decode to produce the soft output, decoder->decoder handoff of the assigned
    region, strong decode, handoff back); otherwise it pays tau_weak alone. The paper
    shows gamma_switch decays exponentially with code distance, so it is a knob here.
    Expected latency: (1-gamma)*tau_weak + gamma*(tau_weak + 2*handoff + tau_strong).

    This answers "is switching worth it" at the latency/queueing level: both decoders'
    own latency models are used per job, and the inter-decoder message passing is an
    explicit, separately-priced hop (`handoff_us`, default = the t_dd link of
    arXiv:2511.10633 Table 2; an FPGA->GPU link can be priced differently).

    What it does NOT model is the double-window interaction (the strong decoder taking
    r_strong = r_com + 2*r_buf rounds while the weak stream pauses and resumes) -- that
    needs the DoubleWindowScheme (schemes.py, currently a documented stub) plus the
    DecoderRouter escalation path. The switch decision is drawn once per job in latency()
    and recorded on job.hint, so decode() reports the same path."""
    def __init__(self, weak: "Decoder", strong: "Decoder", gamma_switch: float,
                 handoff_us: float = 0.5, seed: int = 0):
        """Hold the two decoders, the switch probability, and the handoff link latency."""
        import random
        self.weak = weak
        self.strong = strong
        self.gamma_switch = gamma_switch
        self.handoff = us(handoff_us)
        self.rng = random.Random(seed)
        self.switches = 0                      # diagnostic: how many jobs escalated

    def latency(self, job: DecodeJob) -> int:
        """Weak latency, plus (with probability gamma_switch) handoffs + strong latency."""
        lat = self.weak.latency(job)
        if self.rng.random() < self.gamma_switch:
            job.hint = "strong"                # remember the path so decode() agrees
            self.switches += 1
            lat += 2 * self.handoff + self.strong.latency(job)
        return lat

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Decode via the path latency() chose; record which side answered as soft output
        (1.0 = weak confident, 0.0 = escalated -- stand-ins until real soft outputs)."""
        if job.hint == "strong":
            res = self.strong.decode(job)
            res.soft_output = 0.0
        else:
            res = self.weak.decode(job)
            res.soft_output = 1.0
        return res


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
        return DecodeResult(job.op_id, job.window_id, correction=None, logical_value=None)
 
