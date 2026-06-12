"""Certification of the SLIDING window scheme against its papers (single decoder).

Two pinned results:

1. The window-chain recurrence (arXiv:2511.10633's sequential-window timing): with one
   decoder unit and a zero-latency fabric, every window's completion time must satisfy
       finish(k) = max(data(k), finish(k-1)) + decode(k)
   EXACTLY (integers, no tolerance) -- the sliding analog of the Eq. 5 certification.

2. The keep-pace boundary, Eq. 7 of arXiv:2510.25222 (same onset SWIPER reports as
   r > 0.5 for d/d windows): the chain's lag behind the data stays BOUNDED iff
       tau_dec * (r_com + r_buf) <= tau_gen * r_com ,
   and above the boundary the lag grows by exactly (decode - r_com * tau_gen) per
   window. Tested on d/d windows AND on r_com != r_buf to pin the general formula.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from qecsim.config import us
from qecsim.controllers import ModularController
from qecsim.frontends.circuit import CircuitFrontend
from qecsim.message import DecodeResult, Operation
from qecsim.wiring import build_and_run

TAU_GEN_US = 1.1


class PerRoundDecoder:
    """Decode time = tau_dec per round in the window."""
    def __init__(self, tau_us):
        self.tau_us = tau_us

    def latency(self, job):
        return us(job.n_rounds * self.tau_us)

    def decode(self, job):
        return DecodeResult(job.op_id, job.window_id)


class WindowedCode:
    """Minimal CodeModel with independently chosen commit/buffer sizes, so Eq. 7 can
    be tested beyond the surface code's fixed r_com = r_buf = d."""
    def __init__(self, d, commit, buffer):
        self.d, self._commit, self._buffer = d, commit, buffer

    @property
    def name(self):
        return f"windowed test code (C={self._commit}, B={self._buffer})"

    @property
    def distance(self):
        return self.d

    def rounds_per_logical_cycle(self):
        return self.d

    def rounds_per_op(self):
        return self.d

    def commit_rounds(self):
        return self._commit

    def buffer_rounds(self):
        return self._buffer

    def spatial_nodes(self, num_patches):
        return max(1, num_patches) * self.d * self.d

    def syndrome_bits_per_round(self, num_patches):
        return max(1, num_patches) * self.d * self.d


def _run_memory(tau_us, commit, buffer, n_windows=20):
    """One long memory operation, sliding windows, one decoder, zero-latency links
    (so the recurrence has no fabric terms and Eq. 7 appears in its pure form)."""
    rounds = n_windows * commit                  # R a multiple of C: uniform windows
    op = Operation(0, "M(q0)", (0,), clifford=True)
    ops = CircuitFrontend([op]).build()
    r = build_and_run(ops, num_units=1, code=WindowedCode(3, commit, buffer),
                      rounds_per_op=rounds, round_us=TAU_GEN_US,
                      decoder=PerRoundDecoder(tau_us),
                      make_controller=lambda e: ModularController(
                          e, t_qc=0, t_cd=0, t_dd=0, t_do=0, t_oc=0, t_cq=0,
                          log_syndromes=False),
                      verbose=False)
    return r["cluster"], rounds


def _reference_finish_times(cluster, rounds, tau_us):
    """The recurrence, computed with the simulator's own tick arithmetic."""
    g = us(TAU_GEN_US)
    finish, out = 0, {}
    for k in range(cluster.nwin[0]):
        w = cluster.windows[(0, k)]
        data = g * min(w.buffer_hi, rounds)      # buffer past stream end: data = last round
        decode = us(w.n_rounds * tau_us)
        finish = max(data, finish) + decode
        out[k] = finish
    return out


def test_window_chain_matches_the_recurrence_exactly():
    """finish(k) = max(data(k), finish(k-1)) + decode(k), to the tick, on both sides
    of the keep-pace boundary."""
    for tau in (0.3 * TAU_GEN_US, 0.8 * TAU_GEN_US):
        cluster, rounds = _run_memory(tau, commit=3, buffer=3)
        expected = _reference_finish_times(cluster, rounds, tau)
        for k, t_expected in expected.items():
            assert cluster.windows[(0, k)].t_done == t_expected, \
                f"tau={tau}: window {k} done at {cluster.windows[(0, k)].t_done}, " \
                f"recurrence says {t_expected}"


def _lags(cluster, rounds):
    """Each window's lag behind its own data (finish - data-complete time)."""
    g = us(TAU_GEN_US)
    return [cluster.windows[(0, k)].t_done - g * min(cluster.windows[(0, k)].buffer_hi,
                                                     rounds)
            for k in range(cluster.nwin[0])]


def test_eq7_boundary_with_d_d_windows():
    """r_com = r_buf = d: keep-pace iff tau_dec <= tau_gen/2 (SWIPER's r > 0.5 onset).
    At or below the boundary the lag is the same for every steady-state window; above
    it the lag grows by exactly (decode - stride) per window."""
    C = B = 3
    for tau_frac in (0.3, 0.5):                  # below and EXACTLY AT the boundary
        cluster, rounds = _run_memory(tau_frac * TAU_GEN_US, C, B)
        lag = _lags(cluster, rounds)
        assert lag[5] == lag[10] == lag[-2], f"tau/g={tau_frac}: lag should be flat"
    for tau_frac in (0.6, 0.9):                  # above the boundary
        cluster, rounds = _run_memory(tau_frac * TAU_GEN_US, C, B)
        lag = _lags(cluster, rounds)
        step = us((C + B) * tau_frac * TAU_GEN_US) - us(TAU_GEN_US) * C
        assert lag[10] - lag[9] == step and lag[-2] - lag[-3] == step, \
            f"tau/g={tau_frac}: lag should grow by decode - stride each window"


def test_eq7_boundary_generalizes_beyond_d_d_windows():
    """The general form tau_dec * (C+B) <= tau_gen * C: a bigger commit region tolerates
    a SLOWER decoder (C=9, B=3 -> boundary at 0.75 tau_gen), which is exactly the
    paper's reason r_com is a tunable knob (its Eq. 15 picks r_com from tau_weak)."""
    C, B = 9, 3                                  # boundary: tau <= 0.75 tau_gen
    cluster, rounds = _run_memory(0.7 * TAU_GEN_US, C, B)   # below: must keep pace
    lag = _lags(cluster, rounds)
    assert lag[5] == lag[10] == lag[-2]
    cluster, rounds = _run_memory(0.8 * TAU_GEN_US, C, B)   # above: must fall behind
    lag = _lags(cluster, rounds)
    step = us((C + B) * 0.8 * TAU_GEN_US) - us(TAU_GEN_US) * C
    assert lag[10] - lag[9] == step
    # and the d/d rule would have called 0.7 a FALL-BEHIND (0.7 > 0.5): the general
    # formula, not the special case, is what the simulator reproduces
    assert 0.7 > 0.5