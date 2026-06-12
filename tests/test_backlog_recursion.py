"""Replication of the backlog recursion (arXiv:2510.25222 Eq. 5 / Terhal's argument).

Under NAIVE (batch) decoding with one decoder, the rounds accumulated between
consecutive feedback gates obey
    r_i = (T_comm + tau_dec * r_{i-1}) / tau_gen + rop ,
converging to (rop + T_comm/tau_gen) / (1 - f) for f = tau_dec/tau_gen < 1 and
diverging geometrically with ratio f for f > 1. The mechanism (Terhal, RMP 87, 307,
Sec III.B): the data a patch generates WHILE WAITING for a decode must itself be
processed before the next feedback -- so waiting creates work, which creates waiting.

In qecsim this emerges from one rule: a patch's idle stretch joins the next op's
batch window (chip.idle_rounds_by_patch -> cluster.prepend_idle_rounds, honored by
NaiveOnlineScheme.batches_idle_rounds_into_next_op). These tests pin the simulated
r_i (BacklogTrajectory) to the closed form.

TOLERANCE: rounds are discrete in the simulator, continuous in the formula (the paper
"ignore[s] such rounding operations"). Each gate's idle count differs from the
continuous wait by < 1 round, and that error compounds by f per gate, so
|sim - formula| < sum_j f^j < 1/(1-f) for f < 1, and < (f^i - 1)/(f - 1) for f > 1.
The tests assert within exactly that envelope; gate 1, which involves no idle
feedback yet, must match to the TICK."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from qecsim.config import us
from qecsim.frontends.circuit import CircuitFrontend
from qecsim.message import DecodeResult, Operation
from qecsim.metrics import BacklogTrajectory
from qecsim.schemes import NaiveOnlineScheme, SlidingWindowScheme
from qecsim.wiring import build_and_run

TAU_GEN_US = 1.1                       # syndrome round time
D = 3
ROP = 9 * D                            # the paper's rop = 9d rounds per gate
NGATES = 15
T_COMM_US = 0.15 + 2.0 + 1.0 + 4.0 + 0.15   # qc+cd (in) + do + oc+cq (back): Table 2


class PerRoundDecoder:
    """Decode time = tau_dec per round in the window (the paper's Tdec(r))."""
    def __init__(self, tau_us):
        self.tau_us = tau_us

    def latency(self, job):
        return us(job.n_rounds * self.tau_us)

    def decode(self, job):
        return DecodeResult(job.op_id, job.window_id)


def _t_gate_chain(n=NGATES):
    """n teleported T gates on one patch, each gated on the previous decode."""
    return CircuitFrontend([
        Operation(i, f"T{i}", (0,), clifford=False,
                  gated_by=(i - 1 if i > 0 else None))
        for i in range(n)
    ]).build()


def _simulate(f, scheme):
    r = build_and_run(_t_gate_chain(), num_units=1, d=D, rounds_per_op=ROP,
                      round_us=TAU_GEN_US, scheme=scheme,
                      decoder=PerRoundDecoder(f * TAU_GEN_US),
                      max_idle_rounds=100_000,        # waits far exceed the default cap
                      make_metrics=lambda e, c, ch, fa: [BacklogTrajectory(ch)],
                      verbose=False)
    return BacklogTrajectory(r["chip"]).rows()


def _formula(i, f):
    """Closed form of Eq. 5 with r_0 = rop (gate 0 accumulated no waiting rounds)."""
    a = ROP + T_COMM_US / TAU_GEN_US
    return (f ** i) * ROP + (1 - f ** i) / (1 - f) * a


def test_first_wait_is_exact_to_the_tick():
    """Gate 1 involves no idle feedback yet: wait = T_comm + tau_dec * rop, exactly."""
    for f in (0.4, 0.7, 0.9, 1.1):
        rows = _simulate(f, NaiveOnlineScheme())
        assert rows[0]["wait"] == us(T_COMM_US) + us(ROP * f * TAU_GEN_US)


def test_snowball_converges_to_the_eq5_limit():
    """f < 1: r_i follows the recursion gate by gate (within the rounding envelope)
    and settles at (rop + T_comm/tau_gen)/(1 - f)."""
    for f in (0.4, 0.7, 0.9):
        rows = _simulate(f, NaiveOnlineScheme())
        tol = 1.0 / (1.0 - f)
        for i, row in enumerate(rows, start=1):
            assert abs(row["backlog_rounds"] - _formula(i, f)) < tol, \
                f"f={f} gate {i}: sim {row['backlog_rounds']:.1f} vs " \
                f"formula {_formula(i, f):.1f}"
        limit = (ROP + T_COMM_US / TAU_GEN_US) / (1.0 - f)
        # convergence toward the limit goes as f^i, so after NGATES gates the formula
        # ITSELF is still (limit - rop) * f^NGATES away from it -- allow exactly that
        remaining = (limit - ROP) * f ** NGATES
        assert abs(rows[-1]["backlog_rounds"] - limit) < tol + remaining


def test_snowball_diverges_geometrically_above_f_1():
    """f > 1: r_i grows without bound, tracking the formula's geometric growth."""
    f = 1.1
    rows = _simulate(f, NaiveOnlineScheme())
    backlog = [row["backlog_rounds"] for row in rows]
    assert all(b2 > b1 for b1, b2 in zip(backlog, backlog[1:]))   # strictly growing
    for i, b in enumerate(backlog, start=1):
        envelope = (f ** i - 1) / (f - 1)            # compounded rounding error bound
        assert abs(b - _formula(i, f)) < envelope, \
            f"gate {i}: sim {b:.1f} vs formula {_formula(i, f):.1f}"
    assert backlog[-1] > 3 * backlog[0]              # unmistakably diverging


def test_sliding_scheme_keeps_pace_and_stays_flat():
    """The contrast that proves the mechanism is scheme-scoped: sliding windows with
    f below the Eq. 7 bound (tau_dec <= tau_gen/2 for d/d windows) decode the stream
    as it is produced, so the per-gate backlog must NOT grow."""
    rows = _simulate(0.4, SlidingWindowScheme())
    backlog = [row["backlog_rounds"] for row in rows]
    assert max(backlog[1:]) - min(backlog[1:]) < 1.0


def test_idle_absorption_is_inert_for_windowed_schemes():
    """Sliding runs must never see a batch grow: the absorb log line is naive-only."""
    r = build_and_run(_t_gate_chain(3), num_units=1, d=D, rounds_per_op=ROP,
                      round_us=TAU_GEN_US, decoder=PerRoundDecoder(0.4 * TAU_GEN_US),
                      verbose=False)
    assert not any("absorbs" in l for l in r["engine"].log_lines)


def test_round_grid_mode_matches_the_strict_recursion_exactly():
    """The paper itself says Eq. 5 'strictly speaking ... should be rounded up using
    the ceiling function' -- the continuous closed form is its own simplification.
    With gates starting on the round grid (real hardware's clock; SWIPER-SIM models
    time at round granularity), the simulator must match the STRICT integer recursion
    with NO tolerance at every gate:
        seg_{i+1} = (wait_i // tau_gen + 1) + rop,  wait_i = T_comm + tau_dec * seg_i
    where `// + 1` is the next-boundary rule (a release exactly ON a boundary starts
    at the following one -- the chip's documented convention; identical to the paper's
    ceiling everywhere except exact ties). The reference uses the simulator's own tick
    arithmetic (us()), so the comparison is integers against integers."""
    g = us(TAU_GEN_US)
    t_comm = us(0.15) + us(2.0) + us(1.0) + us(4.0) + us(0.15)   # per-link ticks (LinkModel)
    for f in (0.4, 0.7, 0.9, 1.1):
        tau_us = f * TAU_GEN_US
        r = build_and_run(_t_gate_chain(), num_units=1, d=D, rounds_per_op=ROP,
                          round_us=TAU_GEN_US, scheme=NaiveOnlineScheme(),
                          decoder=PerRoundDecoder(tau_us), max_idle_rounds=100_000,
                          gates_start_on_round_boundaries=True, verbose=False)
        windows = r["cluster"].windows
        seg = ROP                                     # gate 0 absorbed no idle rounds
        assert windows[(0, 0)].n_rounds == seg
        for i in range(1, NGATES):
            wait = t_comm + us(seg * tau_us)          # the simulator's exact decode+links
            seg = (wait // g + 1) + ROP               # idle rounds to the boundary + the gate
            assert windows[(i, 0)].n_rounds == seg, \
                f"f={f} gate {i}: sim {windows[(i, 0)].n_rounds} != strict {seg}"
