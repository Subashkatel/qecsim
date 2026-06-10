#==================================================================
# TESTS FOR WINDOWING (dependency seam + parallel A/B scheme)
#==================================================================
from qecsim.codes import SurfaceCodeModel
from qecsim.config import us
from qecsim.decoders import LatencyModelDecoder, PresetLatencyDecoder
from qecsim.layouts import UniformLayout
from qecsim.message import Operation
from qecsim.planner import WindowPlanner
from qecsim.schemes import SlidingWindowScheme, ParallelWindowScheme
from qecsim.wiring import build_and_run


def _memory_op(rounds_unused=None):
    """One single-patch Clifford op -- a quantum-memory stream."""
    op = Operation(0, "M(q0)", (0,), clifford=True)
    op.patches = (0,)
    return [op]


def _plan(scheme, ops, rounds_per_op, d=3):
    planner = WindowPlanner(scheme, UniformLayout(SurfaceCodeModel(d=d)), rounds_per_op)
    return planner.plan(ops)


# ---- structural: the default chain is unchanged by the seam refactor ----------------

def test_sequential_chain_deps_unchanged():
    plan = _plan(SlidingWindowScheme(), _memory_op(), rounds_per_op=11, d=3)
    assert plan.nwin[0] == 4                      # ceil(11/3)
    assert plan.windows[(0, 0)].deps == []
    for k in range(1, 4):
        assert plan.windows[(0, k)].deps == [(0, k - 1)]
    # no leading buffers in the sequential scheme
    assert all(w.start_round == w.commit_lo for w in plan.windows.values())


def test_cross_op_deps_use_entry_and_exit_defaults():
    a = Operation(0, "A", (0,), clifford=True)
    b = Operation(1, "B", (0,), clifford=True)
    a.patches, b.patches = (0,), (0,)
    b.predecessors, a.has_successor = (0,), True
    plan = _plan(SlidingWindowScheme(), [a, b], rounds_per_op=11, d=3)
    assert plan.windows[(1, 0)].deps == [(0, plan.nwin[0] - 1)]


# ---- structural: parallel A/B layout per arXiv:2511.10633 Sec II.4 ------------------

def test_parallel_scheme_layout_and_deps():
    # d=3: C=B=3, period 2C+2B=12. R=15 -> A_0 [1..6] commit [1,3],
    # B_0 commit [4,12], A_1 [10..18] commit [13,15]; no tail (R = A_1.commit_hi).
    plan = _plan(ParallelWindowScheme(), _memory_op(), rounds_per_op=15, d=3)
    assert plan.nwin[0] == 3
    a0, b0, a1 = (plan.windows[(0, k)] for k in range(3))
    assert (a0.start_round, a0.commit_lo, a0.commit_hi, a0.buffer_hi) == (1, 1, 3, 6)
    assert (b0.commit_lo, b0.commit_hi, b0.buffer_hi) == (4, 12, 12)
    assert (a1.start_round, a1.commit_lo, a1.commit_hi, a1.buffer_hi) == (10, 13, 15, 18)
    # layer-A windows are independent; the layer-B window waits on BOTH neighbours
    assert a0.deps == [] and a1.deps == []
    assert sorted(b0.deps) == [(0, 0), (0, 2)]
    # interior windows have the paper's 3d temporal size
    assert b0.n_rounds == 9 and a1.n_rounds == 9


def test_parallel_scheme_tail_window():
    # R=23 leaves rounds 16..23 after A_1's commit -> a tail window depending on A_1 only.
    plan = _plan(ParallelWindowScheme(), _memory_op(), rounds_per_op=23, d=3)
    assert plan.nwin[0] == 4
    tail = plan.windows[(0, 3)]
    assert (tail.commit_lo, tail.commit_hi) == (16, 23)
    assert tail.deps == [(0, 2)]
    # every round 1..R is committed by exactly one window
    committed = []
    for w in plan.windows.values():
        committed += list(range(w.commit_lo, w.commit_hi + 1))
    assert sorted(committed) == list(range(1, 24))


# ---- ACCEPTANCE: reaction tail reproduces gamma_mem = 6d*tau_d(d^2) + hops (Eq. 13) --

def test_parallel_scheme_reaction_matches_eq13():
    d = 3
    beta = 1.2
    alpha = 1e-6 / (d * d) ** beta      # tau_d(d^2) = 1 us per round, Eq. 12 shape
    r = build_and_run(_memory_op(), num_units=4, d=d, rounds_per_op=15, round_us=1.1,
                      decoder=LatencyModelDecoder(d=d, alpha=alpha, beta=beta),
                      scheme=ParallelWindowScheme(), verbose=False)
    tail = r["fully_done"] - r["chip_done"]
    # after the last round: chip->controller->decoders hops, the last layer-A window
    # (3d rounds), the t_dd boundary, the layer-B window (3d rounds), then t_do.
    # This is Eq. 13's two-window 6d*tau_d(d^2) plus the one-way hops (a Clifford memory
    # op pays no t_oc + t_cq return path -- Pauli-frame updates stay in the orchestrator).
    expected = (us(0.15) + us(2.0)            # t_qc + t_cd
                + us(3 * d * 1.0) + us(0.5)   # 3d rounds at 1 us/round + t_dd
                + us(3 * d * 1.0) + us(1.0))  # 3d rounds + t_do
    assert abs(tail - expected) <= 4          # integer-tick rounding only


# ---- backlog vs units sweep: parallelism helps A/B, cannot help the chain -----------

def test_backlog_sweep_parallel_vs_sequential():
    # service (10 us) exceeds both schemes' window inter-arrival (sequential: one window
    # per commit stride = 3.3 us; parallel: ~2 windows per 12-round period = ~6.6 us), so
    # ONE unit backlogs in both cases -- the question is whether extra units help.
    def run(scheme, units):
        r = build_and_run(_memory_op(), num_units=units, d=3, rounds_per_op=63,
                          round_us=1.1, decoder=PresetLatencyDecoder(10.0),
                          scheme=scheme, verbose=False)
        peak_q = max((q for _, q in r["cluster"].queue_log), default=0)
        return r["fully_done"], peak_q

    seq = {u: run(SlidingWindowScheme(), u) for u in (1, 2, 4)}
    par = {u: run(ParallelWindowScheme(), u) for u in (1, 2, 4)}
    # the sequential chain cannot use extra units: one op's windows decode one at a time
    assert seq[4][0] == seq[1][0]
    # the parallel scheme converts units into completion time and into backlog relief
    assert par[4][0] < par[1][0]
    assert par[4][1] <= par[1][1]
    # and with units available it beats the chain outright
    assert par[4][0] < seq[4][0]
