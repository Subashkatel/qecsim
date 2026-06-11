"""Syndrome-RAM accounting (cluster.payloads_held / peak_payloads).

The high-water mark used to be recomputed by re-summing the WHOLE payload store on
every arriving round -- O(ops) per round, the dominant cost in large runs. It is now
a running counter: +1 when a payload is stored, minus an op's payloads when its store
is freed. These tests prove the counter agrees with a brute-force recount after every
single arrival, and that the store drains back to zero when the workload completes."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from qecsim.cluster import DecoderCluster
from qecsim.decoders import PresetLatencyDecoder
from qecsim.frontends.circuit import cnot_plus_two_t_circuit, three_cnot_circuit
from qecsim.wiring import build_and_run


class RecountingCluster(DecoderCluster):
    """Recounts the whole store after every arrival (the old, slow way) and keeps its
    own maximum, so the running counter can be checked against ground truth."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.brute_force_peak = 0

    def on_syndrome_arrival(self, payload):
        super().on_syndrome_arrival(payload)
        held = sum(len(frags) for per_op in self.payload_store.values()
                   for frags in per_op.values())
        assert self.payloads_held == held, "running counter drifted from the store"
        self.brute_force_peak = max(self.brute_force_peak, held)


def _run(ops):
    built = {}
    def make_cluster(engine, decoder, scheduler, controller, orchestrator):
        c = RecountingCluster(engine, decoder, scheduler, controller, orchestrator,
                              num_units=2, code_distance=3)
        built["cluster"] = c
        return c
    build_and_run(ops, make_cluster=make_cluster, d=3, rounds_per_op=11,
                  decoder=PresetLatencyDecoder(1.0), verbose=False)
    return built["cluster"]


def test_peak_payloads_matches_brute_force_recount():
    cluster = _run(three_cnot_circuit())
    assert cluster.peak_payloads == cluster.brute_force_peak > 0


def test_accounting_with_gated_ops_and_store_drains_to_zero():
    """Gated T gates exercise idle rounds and late window commits; afterwards every
    op's store has been freed, so an exact counter must read zero."""
    cluster = _run(cnot_plus_two_t_circuit())
    assert cluster.peak_payloads == cluster.brute_force_peak > 0
    assert cluster.payloads_held == 0


def test_round_arriving_after_op_completed_fails_loudly():
    """A payload for an op whose syndrome RAM was already freed (its last window
    committed) means the device emitted more rounds than planned -- the cluster must
    say so, not die on a KeyError or corrupt the running counter."""
    import pytest
    from qecsim.message import SyndromePayload
    cluster = _run(three_cnot_circuit())                  # run to completion
    with pytest.raises(RuntimeError, match="syndrome RAM was freed"):
        cluster.on_syndrome_arrival(SyndromePayload(0, 0, 99))
