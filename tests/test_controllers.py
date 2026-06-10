#==================================================================
# TESTS FOR CONTROLLER-SIDE ROUND PACKAGING
# arXiv:2511.10633 Sec III.1: the controller aggregates per-qubit readout and
# forwards batched packets to the decoders -- so a round arriving in fragments
# (possibly at different times) must ship as ONE packet after its last fragment.
#==================================================================
from qecsim.config import us
from qecsim.controllers import ModularController
from qecsim.engine import Engine
from qecsim.message import SyndromePayload


def _frag(patch, n=2):
    return SyndromePayload(0, patch, 1, n_fragments=n)


def test_staggered_fragments_ship_as_one_packet_after_the_last():
    eng = Engine(verbose=False)
    ctrl = ModularController(eng, log_syndromes=False)
    arrivals = []
    deliver = lambda p: arrivals.append((eng.now, p.patch_id))
    # the device emits the round in two chunks, 0.4 us apart (staggered readout)
    eng.schedule(0, lambda: ctrl.relay_syndrome(_frag(0), deliver))
    eng.schedule(us(0.4), lambda: ctrl.relay_syndrome(_frag(1), deliver))
    eng.run()
    # both fragments arrive TOGETHER, one t_qc + t_cd after the LAST chunk left the chip
    expected = us(0.4) + us(0.15) + us(2.0)
    assert arrivals == [(expected, 0), (expected, 1)]


def test_packaging_cost_is_priced_per_packet():
    eng = Engine(verbose=False)
    ctrl = ModularController(eng, log_syndromes=False, t_pack=us(0.3))
    arrivals = []
    eng.schedule(0, lambda: ctrl.relay_syndrome(_frag(0), lambda p: arrivals.append(eng.now)))
    eng.schedule(0, lambda: ctrl.relay_syndrome(_frag(1), lambda p: arrivals.append(eng.now)))
    eng.run()
    assert arrivals == [us(0.15) + us(0.3) + us(2.0)] * 2


def test_whole_round_payloads_take_the_original_path():
    # n_fragments=1 (every default device): no buffering, no t_pack -- two plain hops
    eng = Engine(verbose=False)
    ctrl = ModularController(eng, log_syndromes=False, t_pack=us(9.9))
    arrivals = []
    eng.schedule(0, lambda: ctrl.relay_syndrome(_frag(0, n=1),
                                                lambda p: arrivals.append(eng.now)))
    eng.run()
    assert arrivals == [us(0.15) + us(2.0)]
