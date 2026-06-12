"""Per-window decoding problems (gap #7, phase R1): the slicing must be paper-exact.

Pinned to the sources in docs/DESIGN-real-window-decoding.md:
- ownership partition (Skoric's crossing-edge commit rule / QUITS's column cursor):
  every fault committed by exactly ONE window;
- open interior time boundaries (Tan): cut faults become single-detector columns;
- artificial-defect handoff (all three papers, one mechanism): a committed fault's
  beyond-commit flips cancel the defects the next window sees;
- the certification anchor (Skoric App C): windowed decoding with buffer d matches
  whole-history decoding accuracy.

Requires stim + pymatching (skipped where unavailable, like the other adapters)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")

from qecsim.adapters.window_problems import (build_window_problems,
                                             decode_windowed,
                                             detector_error_model_to_faults,
                                             matching_window_decoder)
from qecsim.codes import SurfaceCodeModel
from qecsim.schemes import SlidingWindowScheme


def _memory_circuit(d=3, rounds=12, p=0.003):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=p, after_reset_flip_probability=p,
        before_measure_flip_probability=p, before_round_data_depolarization=p)


def _plan(circuit, d=3):
    """The REAL scheme's window plan over the circuit's detector layers
    (layer t = round t+1), exactly as the cluster would plan it."""
    n_layers = 1 + max(int(c[-1]) for c in
                       circuit.get_detector_coordinates().values())
    return SlidingWindowScheme().plan_windows(0, n_layers, SurfaceCodeModel(d=d))


def test_fault_conversion_merges_duplicates_with_the_standard_rule():
    """p (+) q = p(1-q) + q(1-p), the BeliefMatching/QUITS convention."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1
        error(0.2) D0 D1
        error(0.3) D1 D2 L0
    """)
    det_sets, obs_sets, priors = detector_error_model_to_faults(dem)
    assert len(det_sets) == 2
    i = det_sets.index((0, 1))
    assert priors[i] == pytest.approx(0.1 * 0.8 + 0.2 * 0.9)
    j = det_sets.index((1, 2))
    assert obs_sets[j] == (0,)


def test_composite_errors_split_into_matchable_components():
    """Stim's `^`-separated decompositions become separate <=2-detector faults, each
    carrying the parent probability (PyMatching's own convention)."""
    dem = stim.DetectorErrorModel("error(0.25) D0 D1 ^ D2 D3 L0")
    det_sets, obs_sets, priors = detector_error_model_to_faults(dem)
    assert sorted(det_sets) == [(0, 1), (2, 3)]
    assert priors == [0.25, 0.25]
    # and on the real circuit, EVERY column is matchable
    circuit = _memory_circuit()
    sets, _, _ = detector_error_model_to_faults(
        circuit.detector_error_model(decompose_errors=True))
    assert max(len(s) for s in sets) <= 2


def test_every_fault_is_owned_by_exactly_one_window():
    """The commit partition: each fault decided once, none lost (Skoric's rule)."""
    circuit = _memory_circuit()
    problems = build_window_problems(circuit, _plan(circuit))
    det_sets, _, _ = detector_error_model_to_faults(
        circuit.detector_error_model(decompose_errors=True))
    owned_total = sum(int(p.owned.sum()) for p in problems)
    assert owned_total == len(det_sets)


def test_interior_windows_have_open_time_boundaries():
    """A fault straddling a window's edge appears as a single-detector column -- the
    boundary edge Tan's imaginary detectors formalize."""
    circuit = _memory_circuit()
    problems = build_window_problems(circuit, _plan(circuit))
    interior = problems[1]
    assert (interior.check.sum(axis=0) == 1).any()


def test_single_crossing_fault_round_trips_exactly():
    """THE mechanics test: a single fault that crosses a commit boundary must be
    committed by its owning window, hand its beyond-commit flips forward as
    artificial defects, and the windowed pass must reproduce the fault's observable
    flips exactly -- with every handed-forward defect consumed."""
    circuit = _memory_circuit()
    problems = build_window_problems(circuit, _plan(circuit))
    det_sets, obs_sets, _ = detector_error_model_to_faults(
        circuit.detector_error_model(decompose_errors=True))
    w0 = problems[0]
    crossing_cols = [c for c in w0.future_flips if w0.owned[c]]
    assert crossing_cols, "no boundary-crossing fault found in window 0"
    decode = matching_window_decoder()
    n_dets = circuit.num_detectors
    checked = 0
    for col in crossing_cols[:5]:
        # rebuild the GLOBAL detection events of exactly this fault
        in_window = set(np.asarray(w0.detector_ids)[w0.check[:, col] > 0])
        beyond = set(w0.future_flips[col])
        events = np.zeros(n_dets, dtype=np.uint8)
        for d in in_window | beyond:
            events[d] = 1
        predicted = decode_windowed(problems, events, decode)
        # which fault is this, globally? find it by its full detector set
        full = tuple(sorted(in_window | beyond))
        expected = np.zeros(circuit.num_observables, dtype=np.uint8)
        for o in obs_sets[det_sets.index(full)]:
            expected[o] = 1
        assert (predicted == expected).all(), f"fault {full} mis-roundtripped"
        checked += 1
    assert checked > 0


def test_windowed_accuracy_matches_global_decoding():
    """Skoric Appendix C, the published anchor: with buffer = d, sliding-window
    decoding shows 'no noticeable increase in logical error rate' over decoding the
    whole history at once. Fixed seed -> deterministic counts."""
    pymatching = pytest.importorskip("pymatching")
    circuit = _memory_circuit(d=3, rounds=12, p=0.003)
    problems = build_window_problems(circuit, _plan(circuit))
    shots = 2000
    dets, obs = circuit.compile_detector_sampler(seed=11).sample(
        shots, separate_observables=True)
    global_m = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    global_pred = global_m.decode_batch(dets)
    decode = matching_window_decoder()
    windowed_pred = np.array([decode_windowed(problems, dets[i], decode)
                              for i in range(shots)])
    agree = float((windowed_pred == global_pred).all(axis=1).mean())
    ler_global = float((global_pred != obs).any(axis=1).mean())
    ler_windowed = float((windowed_pred != obs).any(axis=1).mean())
    assert agree > 0.97, f"windowed disagrees with global too often: {agree}"
    # 'no noticeable increase': allow binomial wiggle on 2000 shots, nothing more
    assert ler_windowed <= ler_global + 2 * (ler_global / shots) ** 0.5 + 0.005, \
        f"windowed LER {ler_windowed} vs global {ler_global}"
