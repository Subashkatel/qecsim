"""Typed decoder-unit pools (cluster.unit_pools).

Regression for a resource-accounting inaccuracy: every decode used to draw from one
anonymous unit pool, so a slow strong-decoder job could occupy -- and make ready weak
windows queue behind -- a unit that models weak hardware. Each pool now owns its units
AND its own ready queue, picked by job.hint at enqueue time (arXiv:2510.25222 Fig 1:
weak = FPGA/ASIC, strong = CPU/GPU). No pools configured = one "default" pool,
byte-identical to before."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from qecsim.cluster import DecoderCluster
from qecsim.config import us
from qecsim.decoders import PresetLatencyDecoder
from qecsim.engine import Engine
from qecsim.frontends.circuit import cnot_plus_two_t_circuit
from qecsim.schedulers import FifoScheduler
from qecsim.wiring import build_and_run


def _run(**kw):
    r = build_and_run(cnot_plus_two_t_circuit(), d=3, rounds_per_op=11,
                      decoder=PresetLatencyDecoder(1.0), verbose=False, **kw)
    return r["engine"].log_lines


def test_default_pool_matches_plain_num_units():
    """unit_pools={"default": n} is exactly num_units=n -- the byte-identical guarantee."""
    assert _run(num_units=2) == _run(unit_pools={"default": 2})


def test_idle_extra_pool_changes_nothing():
    """A strong pool no job targets must not alter the trace in any way."""
    assert _run(num_units=2) == _run(unit_pools={"default": 2, "strong": 1})


def test_strong_jobs_queue_on_their_own_unit():
    """Two strong jobs and one default job, one unit each: the default job runs at t=0
    even though the strong unit is busy, and the second strong job waits for the FIRST
    STRONG job -- not for the default unit."""
    engine = Engine(verbose=False)
    cluster = DecoderCluster(engine, PresetLatencyDecoder(10.0), FifoScheduler(),
                             None, None, num_units=1, code_distance=3,
                             unit_pools={"default": 1, "strong": 1})
    done = {}
    cluster.submit_decode(6, lambda: done.update(A=engine.now), label="A", hint="strong")
    cluster.submit_decode(6, lambda: done.update(B=engine.now), label="B", hint="strong")
    cluster.submit_decode(6, lambda: done.update(C=engine.now), label="C")
    engine.run()
    assert done["A"] == us(10.0)      # started at t=0 on the strong unit
    assert done["C"] == us(10.0)      # started at t=0 on the default unit, unblocked
    assert done["B"] == us(20.0)      # queued behind A on the strong unit only
    assert any("strong units free now 0" in l for l in engine.log_lines)


def test_unknown_hint_runs_on_the_default_pool():
    """A hint that names no pool is only a router hint -- the job uses default units."""
    engine = Engine(verbose=False)
    cluster = DecoderCluster(engine, PresetLatencyDecoder(10.0), FifoScheduler(),
                             None, None, num_units=1, code_distance=3)
    done = {}
    cluster.submit_decode(6, lambda: done.update(A=engine.now), label="A", hint="gpu")
    engine.run()
    assert done["A"] == us(10.0)


def test_pool_validation_fails_loudly():
    args = (Engine(verbose=False), PresetLatencyDecoder(1.0), FifoScheduler(), None, None)
    with pytest.raises(ValueError, match='"default" pool'):
        DecoderCluster(*args, num_units=1, code_distance=3, unit_pools={"strong": 1})
    with pytest.raises(ValueError, match="at least 1 unit"):
        DecoderCluster(*args, num_units=1, code_distance=3,
                       unit_pools={"default": 1, "strong": 0})


def test_units_conserved_when_a_decoder_mutates_the_hint_mid_flight():
    """SwitchingDecoder sets job.hint='strong' DURING latency() -- after the job already
    dispatched on the default pool. The done-path must free the pool the job actually
    ran on (job.pool), not the pool its mutated hint now names."""
    from qecsim.decoders import SwitchingDecoder
    for seed in range(5):
        sw = SwitchingDecoder(PresetLatencyDecoder(1.0), PresetLatencyDecoder(10.0),
                              gamma_switch=0.5, seed=seed)
        r = build_and_run(cnot_plus_two_t_circuit(), d=3, rounds_per_op=11, decoder=sw,
                          unit_pools={"default": 2, "strong": 1}, verbose=False)
        cluster = r["cluster"]
        assert cluster.pool_free == cluster.unit_totals, \
            f"seed {seed}: a unit leaked into the wrong pool"


def test_pools_with_deadline_scheduler_complete_and_conserve():
    """EDF sorts each pool's queue independently; every job finishes and every pool's
    units all come back."""
    from qecsim.schedulers import EarliestDeadlineScheduler
    engine = Engine(verbose=False)
    cluster = DecoderCluster(engine, PresetLatencyDecoder(7.0),
                             EarliestDeadlineScheduler(), None, None,
                             num_units=1, code_distance=3,
                             unit_pools={"default": 2, "strong": 2})
    done = []
    for i in range(6):
        cluster.submit_decode(6, lambda i=i: done.append(i), label=f"s{i}",
                              hint="strong", deadline=us(100 - i))
        cluster.submit_decode(6, lambda i=i: done.append(i + 10), label=f"d{i}",
                              deadline=us(100 - i))
    engine.run()
    assert len(done) == 12
    assert cluster.pool_free == cluster.unit_totals
