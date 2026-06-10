#==================================================================
# TESTS FOR FACTORIES (continuous production + per-state provenance)
#==================================================================
import pytest

from qecsim.config import us
from qecsim.engine import Engine
from qecsim.factories import (DistillationFactory, DistillLevel,
                              MultiLevelDistillationFactory)
from qecsim.metrics import MagicStateLatency


class ImmediateService:
    """A DecodeService stub: every correction decode completes instantly."""
    def submit_decode(self, n_rounds, on_done, label="", deadline=None,
                      code=None, spatial_nodes=None):
        on_done()


def test_continuous_requires_capacity():
    eng = Engine(verbose=False)
    with pytest.raises(ValueError):
        DistillationFactory(eng, 1, us(10), ImmediateService(), corr_rounds=1,
                            production="continuous")
    with pytest.raises(ValueError):
        DistillationFactory(eng, 1, us(10), ImmediateService(), corr_rounds=1,
                            production="freerun")


def test_continuous_fills_buffer_and_halts():
    eng = Engine(verbose=False)
    f = DistillationFactory(eng, num_units=2, cycle_ticks=us(10),
                            decode_service=ImmediateService(), corr_rounds=1, n_corr=2,
                            production="continuous", buffer_capacity=3)
    eng.run()                                  # free-runs from t=0, no request needed
    assert f.produced == 3 and f.store == 3    # filled to capacity, then HALTED
    # consuming a state re-opens a buffer slot and production resumes
    delivered = []
    f.request(0, lambda: delivered.append(True))
    eng.run()
    assert delivered == [True]
    assert f.produced == 4 and f.store == 3    # refilled the slot just taken


def test_demand_mode_produces_nothing_unasked():
    eng = Engine(verbose=False)
    f = DistillationFactory(eng, num_units=2, cycle_ticks=us(10),
                            decode_service=ImmediateService(), corr_rounds=1, n_corr=2)
    eng.run()
    assert f.produced == 0                     # demand-driven: idle without requests


def test_state_trace_provenance():
    eng = Engine(verbose=False)
    f = DistillationFactory(eng, num_units=1, cycle_ticks=us(10),
                            decode_service=ImmediateService(), corr_rounds=1, n_corr=2,
                            return_ticks=us(2.0))
    f.request(0, lambda: None)
    eng.run()
    assert len(f.traces) == 1
    tr = f.traces[0]
    assert tr.t_phys_done - tr.t_distill_start == us(10)   # the distillation cycle
    assert tr.t_corr_done == tr.t_phys_done                # immediate correction decodes
    assert tr.t_released - tr.t_corr_done == us(2.0)       # the return trip
    assert tr.t_delivered == tr.t_released                 # a consumer was waiting


def test_magic_state_latency_metric():
    eng = Engine(verbose=False)
    f = DistillationFactory(eng, num_units=1, cycle_ticks=us(10),
                            decode_service=ImmediateService(), corr_rounds=1, n_corr=2)
    f.request(0, lambda: None)
    eng.run()
    res = MagicStateLatency(f).result()
    assert res["distill"] == {"mean": us(10), "max": us(10), "n": 1}
    assert res["total"]["n"] == 1


def test_multilevel_continuous_fills_top_buffer():
    eng = Engine(verbose=False)
    f = MultiLevelDistillationFactory(
        eng, [DistillLevel(units=1, d=3)], W_ticks=us(1.0), M=2, N=1,
        prep_units=4, production="continuous", buffer_capacity=2)
    eng.run()
    assert f.buffer[1] == 2 and f.produced[1] == 2         # filled to capacity, halted
