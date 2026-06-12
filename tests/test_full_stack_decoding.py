"""Full-stack real decoding (gap #7, phase R3): real syndromes through the ENTIRE
simulator -- StimDevice -> controller -> cluster windows -> PyMatchingDecoder ->
orchestrator -- must reproduce the offline reference workbench exactly.

The acceptance chain (docs/DESIGN-real-window-decoding.md):
- engine prediction == decode_windowed() offline reference, per shot, EXACTLY
  (same models, same matchings, so any deviation is a wiring bug);
- engine prediction == global whole-history decoding, per shot (Skoric App C's
  buffer-d anchor, here realized as exact agreement at these sizes);
- artificial defects genuinely flow between windows (non-empty boundary_defects)
  and are consumed (agreement could not hold otherwise);
- the device's folded round convention: chip round r carries stim layer t = r-1,
  closing layers fold into the last round (round = min(t+1, R)).

Requires stim + pymatching (skipped where unavailable)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pymatching = pytest.importorskip("pymatching")

from qecsim.message import Operation
from qecsim.wiring import build_and_run
from qecsim.adapters.stim_device import StimDevice
from qecsim.adapters.pymatching_decoder import PyMatchingDecoder
from qecsim.adapters.window_error_models import (build_window_error_models,
                                             decode_windowed,
                                             matching_window_decoder)
from qecsim.schemes import SlidingWindowScheme
from qecsim.codes import SurfaceCodeModel
from qecsim.planner import FixedRounds


D, ROUNDS, P = 3, 12, 0.003


class _ZeroLatency:
    def latency(self, job):
        return 1


def _circuit():
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=D, rounds=ROUNDS,
        after_clifford_depolarization=P, after_reset_flip_probability=P,
        before_measure_flip_probability=P, before_round_data_depolarization=P)


def _run_engine_shot(circuit, device, decoder):
    op = Operation(id=1, name="memory", qubits=(0,), clifford=True, circuit=circuit)
    res = build_and_run(ops=[op], num_units=4, d=D,
                        rounds_policy=FixedRounds(ROUNDS),
                        code=SurfaceCodeModel(d=D), scheme=SlidingWindowScheme(),
                        device=device, decoder=decoder, verbose=False)
    return res["cluster"].op_results[1]


def test_stim_device_round_alignment():
    """Round 1 carries layer t=0 (the init detectors -- the bits the old off-by-one
    dropped); the last round carries its own layer AND the folded closing layer."""
    circuit = _circuit()
    device = StimDevice(seed=3)
    op = Operation(id=1, name="memory", qubits=(0,), clifford=True, circuit=circuit)
    device.begin_operation(op)
    coords = circuit.get_detector_coordinates()
    layer = {}
    for det, c in coords.items():
        layer.setdefault(int(c[-1]), []).append(det)
    r1 = device.round_payload(op, 1)
    assert len(r1.bits) == len(layer[0])
    last = device.round_payload(op, ROUNDS)
    assert len(last.bits) == len(layer[ROUNDS - 1]) + len(layer[ROUNDS])
    # nothing beyond the chip's rounds, nothing at round 0
    assert len(device.round_payload(op, ROUNDS + 1).bits) == 0
    assert len(device.round_payload(op, 0).bits) == 0
    # every detector bit is emitted exactly once across rounds 1..R
    total = sum(len(device.round_payload(op, r).bits) for r in range(1, ROUNDS + 1))
    assert total == circuit.num_detectors


def test_engine_matches_offline_reference_and_global_exactly():
    """THE R3 gate: per shot, the engine's decoded logical value equals the offline
    decode_windowed reference (exact -- same construction end to end) and the global
    whole-history decode; LERs are equal by implication. Defects must really flow."""
    circuit = _circuit()
    # offline reference built with the engine's own folded round convention
    coords = circuit.get_detector_coordinates()
    folded = {det: min(int(c[-1]) + 1, ROUNDS) for det, c in coords.items()}
    plan = [(lo, hi, min(b, ROUNDS)) for lo, hi, b in
            SlidingWindowScheme().plan_windows(0, ROUNDS, SurfaceCodeModel(d=D))]
    ref_models = build_window_error_models(circuit, plan, detector_rounds=folded)
    ref_inner = matching_window_decoder()
    global_m = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))

    defect_bits = 0

    class CountingDecoder(PyMatchingDecoder):
        def decode(self, job):
            nonlocal defect_bits
            r = super().decode(job)
            if r.boundary_defects:
                defect_bits += sum(sum(m) for m in r.boundary_defects.values())
            return r

    device = StimDevice(seed=11)
    shots = 150
    for s in range(shots):
        pred_engine = _run_engine_shot(circuit, device, CountingDecoder(_ZeroLatency()))
        shot = device._dets[1]
        pred_offline = int(decode_windowed(ref_models, shot, ref_inner)[0])
        pred_global = int(global_m.decode(shot)[0])
        assert pred_engine == pred_offline, f"shot {s}: engine != offline reference"
        assert pred_engine == pred_global, f"shot {s}: engine != global decode"
    assert defect_bits > 0, "no artificial defects ever crossed a commit boundary"


def test_timing_only_ops_still_run():
    """An op without a circuit keeps dem=None and decodes as a timing-only stub --
    the real-decoding wiring must not break the timing pipeline."""
    op = Operation(id=1, name="timing", qubits=(0,), clifford=True)
    res = build_and_run(ops=[op], num_units=4, d=D,
                        rounds_policy=FixedRounds(ROUNDS),
                        code=SurfaceCodeModel(d=D), scheme=SlidingWindowScheme(),
                        decoder=PyMatchingDecoder(_ZeroLatency()), verbose=False)
    assert res["cluster"].op_results == {}        # no logical value, but it completed
    assert len(res["cluster"].committed_windows) == res["cluster"].total_windows
