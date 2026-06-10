#==================================================================
# TESTS FOR BOUNDARY DATA (artificial defects) + PER-PATCH PAYLOADS
# Mechanism per Skoric et al. arXiv:2209.08552 Sec I.B/Fig. 2: committing
# correction chains that cross out of the commit region creates artificial
# defects just outside it; the dependent window must include them.
#==================================================================
from qecsim.cluster import DecoderCluster
from qecsim.codes import SurfaceCodeModel
from qecsim.config import us
from qecsim.controllers import ModularController
from qecsim.decoders import PresetLatencyDecoder
from qecsim.devices import SyndromeBitDevice
from qecsim.engine import Engine
from qecsim.frontends.circuit import CircuitFrontend
from qecsim.message import DecodeResult, Operation, SyndromePayload
from qecsim.orchestrators import PauliFrameOrchestrator
from qecsim.schedulers import FifoScheduler
from qecsim.wiring import build_and_run


MASK = (1, 0, 1, 0, 1, 0, 1, 0)


class DefectEmittingDecoder:
    """Records every payload each window's decode sees, and (optionally) emits a fixed
    artificial-defect mask at the round just past its commit region -- the place crossing
    chains put defects (arXiv:2209.08552 Fig. 2)."""
    def __init__(self, emit: bool):
        self.emit = emit
        self.seen = {}   # (job_op, window, payload_op, round, patch) -> bits tuple

    def latency(self, job):
        return us(0.1)

    def decode(self, job):
        for p in job.payloads:
            key = (job.op_id, job.window_id, p.operation_id, p.round_index, p.patch_id)
            self.seen[key] = tuple(int(b) for b in (p.bits if p.bits is not None else ()))
        defects = {job.window.commit_hi + 1: list(MASK)} \
            if self.emit and job.window is not None else None
        return DecodeResult(job.op_id, job.window_id, boundary_defects=defects)


def _memory_ops(n=1):
    ops = [Operation(i, f"M{i}(q0)", (0,), clifford=True) for i in range(n)]
    return CircuitFrontend(ops).build()


def _run(ops, emit, device=None):
    dec = DefectEmittingDecoder(emit)
    build_and_run(ops, num_units=2, d=3, rounds_per_op=11, decoder=dec,
                  device=device, verbose=False)
    return dec.seen


def test_artificial_defects_reach_next_window_same_op():
    # sequential scheme, one op: W_k commits rounds 3k+1..3k+3; W_{k-1}'s defects land
    # exactly on W_k's first round (3k+1).
    device = lambda: SyndromeBitDevice(SurfaceCodeModel(d=3), seed=1)
    base = _run(_memory_ops(), emit=False, device=device())
    flipped = _run(_memory_ops(), emit=True, device=device())
    assert set(base) == set(flipped)
    changed = []
    for key, bits in base.items():
        _, k, _, r, _ = key
        if k >= 1 and r == 3 * k + 1:          # the dependent window's boundary round
            assert flipped[key] == tuple(b ^ m for b, m in zip(bits, MASK))
            changed.append(key)
        else:                                   # every other view is untouched -- including
            assert flipped[key] == bits         # the SOURCE window's own buffer view of r
    assert len(changed) == 3                    # W1, W2, W3 each received defects


def test_artificial_defects_cross_op_shift():
    # two chained ops, 11 rounds each: op0's last window commits up to round 11 and emits
    # defects at op0-local round 12 -> shifted to op1-local round 1 on op1's first window.
    device = SyndromeBitDevice(SurfaceCodeModel(d=3), seed=2)
    base = _run(_memory_ops(2), emit=False, device=device)
    device = SyndromeBitDevice(SurfaceCodeModel(d=3), seed=2)
    flipped = _run(_memory_ops(2), emit=True, device=device)
    key = next(k for k in base
               if k[0] == 1 and k[1] == 0 and k[2] == 1 and k[3] == 1)  # op1 W0, round 1
    assert flipped[key] == tuple(b ^ m for b, m in zip(base[key], MASK))
    # op0's own last window reads op1's round 1 as buffer overflow -- WITHOUT the mask
    # (the defects belong to op1's window, and assembly copies never mutate the store)
    overflow_key = (0, 3, 1, 1, key[4])
    assert flipped[overflow_key] == base[overflow_key]


def test_timing_only_payload_becomes_defect_mask():
    # with the default timing-only device (bits=None), an arriving mask IS the data
    flipped = _run(_memory_ops(), emit=True)     # TimingOnlyDevice default
    w1_first = next(k for k in flipped if k[1] == 1 and k[3] == 4)
    assert flipped[w1_first] == MASK


def test_per_patch_fragments_gate_round_arrival():
    # a round with n_fragments=2 only counts as arrived once BOTH patches are in
    eng = Engine(verbose=False)
    cl = DecoderCluster(eng, PresetLatencyDecoder(1.0), FifoScheduler(),
                        ModularController(eng), PauliFrameOrchestrator(eng),
                        num_units=1, code_distance=3)
    op = Operation(0, "CNOT(q0,q1)", (0, 1), clifford=True)
    op.patches = (0, 1)
    cl.register_op(op)
    cl.build_windows()
    cl.on_syndrome_arrival(SyndromePayload(0, 0, 1, n_fragments=2))
    assert cl.rounds_arrived[0] == 0             # half the round is not the round
    cl.on_syndrome_arrival(SyndromePayload(0, 1, 1, n_fragments=2))
    assert cl.rounds_arrived[0] == 1


def test_per_patch_device_end_to_end():
    # a 2-patch op with a per-patch device: every decoded round carries BOTH fragments
    op = Operation(0, "CNOT(q0,q1)", (0, 1), clifford=True)
    dec = DefectEmittingDecoder(emit=False)
    build_and_run(CircuitFrontend([op]).build(), num_units=1, d=3, rounds_per_op=11,
                  decoder=dec,
                  device=SyndromeBitDevice(SurfaceCodeModel(d=3), seed=3, per_patch=True),
                  verbose=False)
    w0_rounds = {(r, p) for (_, k, _, r, p) in dec.seen if k == 0}
    assert {(r, p) for r in range(1, 7) for p in (0, 1)} <= w0_rounds
