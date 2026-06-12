from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# =====================================================================================
# PER-WINDOW DECODING PROBLEMS (gap #7)
#
# A decoder needs two inputs: the measured detector
# bits (which arrive shot by shot) and the ERROR MODEL -- the catalog of everything
# that can go wrong, listing for each fault which detectors it flips, how likely it
# is, and whether it flips the logical answer. stim computes that catalog ONCE for
# the whole circuit (the detector error model, DEM). Windowed decoding never builds
# per-window circuits; it slices the one global catalog into per-window pieces.
# A WindowErrorModel is one such piece: the prepared reference sheet a decoder needs
# to solve window k. It is pure compile-time data -- built before any shot exists,
# shared by every shot; only the detector bits change per shot.
#
# The slicing rules are multi-source verified (docs/DESIGN-real-window-decoding.md,
# incl. the 8-source architecture cross-check in its section 2b):
#   - row-slice the global DEM by detector rounds (Tan arXiv:2209.09219; QUITS
#     spacetime(); Huang & Puri arXiv:2311.03307)
#   - a fault is OWNED (committed) by the FIRST window whose commit region it touches
#     (Skoric arXiv:2209.08552's commit-the-crossing-edges rule; QUITS's advancing
#     column cursor; Bombin arXiv:2303.04846's disjoint commit-region partition);
#     the last window owns everything left (the experiment's true closing boundary).
#     Ownership is what guarantees every fault is decided exactly once.
#   - interior window time boundaries are OPEN: a fault whose other detector was cut
#     out of the slice becomes a single-detector column = a boundary edge (Tan's
#     imaginary detectors, mechanically free here)
#   - a committed fault's detector flips BEYOND the commit region are the artificial
#     defects handed forward (Skoric; Huang & Puri's sigma' = sigma + H*xi; QUITS's
#     window_update) -- the note telling the next window "already explained, ignore"
#
# The slice is CODE-AGNOSTIC: the same construction serves the surface code and
# qLDPC / bivariate-bicycle codes (QUITS validates it for qLDPC). Only the inner
# decoder differs: matching_window_decoder() (PyMatching) for matchable codes, built
# with decompose_errors=True; bposd_window_decoder() (ldpc BP-OSD) for BB/qLDPC,
# built with decompose_errors=False since their faults may flip >2 detectors.
# =====================================================================================


@dataclass(frozen=True)
class WindowErrorModel:
    """One window's decoding problem, sliced from the operation's global DEM.

    Rows are the window's detectors (sorted by global id, which stim orders by time);
    columns are the candidate faults this window may select: every fault touching its
    rows that no earlier window committed."""
    detector_ids: tuple        # global detector ids of the rows, in row order
    commit_hi: int             # last committed round (1-based; round = stim t + 1)
    check: "object"            # uint8 (n_rows, n_cols): fault -> in-window detector flips
    priors: "object"           # float (n_cols,): each fault's probability
    obs: "object"              # uint8 (n_obs, n_cols): fault -> logical observable flips
    owned: "object"            # bool (n_cols,): faults THIS window commits
    future_flips: dict         # owned col -> tuple of GLOBAL detector ids it flips
    #                            beyond commit_hi (the artificial defects handed on)


def detector_error_model_to_faults(dem) -> tuple:
    """The standard DEM -> fault-list conversion (BeliefMatching / QUITS lineage).

    Composite errors (stim's `^`-separated suggested decompositions) are SPLIT into
    their components, each carrying the parent's probability -- the convention
    PyMatching itself applies, required for matchable (<= 2 detectors) columns.
    Identical (detectors, observables) faults merge with p (+) q = p(1-q) + q(1-p).

    Returns (det_sets, obs_sets, priors): parallel lists, one entry per fault."""
    merged: dict = {}                              # (dets, obs) -> prior
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        components, dets, obs = [], [], []
        for t in inst.targets_copy():
            if t.is_separator():
                components.append((tuple(sorted(dets)), tuple(sorted(obs))))
                dets, obs = [], []
            elif t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs.append(t.val)
        components.append((tuple(sorted(dets)), tuple(sorted(obs))))
        for key in components:
            if not key[0]:
                continue                           # component with no detectors
            q = merged.get(key, 0.0)
            merged[key] = q * (1 - p) + p * (1 - q)
    det_sets = [k[0] for k in merged]
    obs_sets = [k[1] for k in merged]
    priors = list(merged.values())
    return det_sets, obs_sets, priors


def build_window_error_models(circuit, plan: list, num_observables: Optional[int] = None,
                          *, decompose_errors: bool = True,
                          detector_rounds: Optional[dict] = None) -> list:
    """Slice an operation's circuit into one WindowErrorModel per planned window.

    `plan` is scheme-style: [(commit_lo, commit_hi, buffer_hi), ...] in 1-based rounds,
    where round r covers the detectors with stim time coordinate t = r - 1. Detectors
    past the last window's buffer (the final data-measurement layer) join the LAST
    window -- the experiment's true closing time boundary (QUITS's special last
    window; Tan's closed final boundary).

    `decompose_errors` mirrors stim's flag: True (default) splits faults into the
    <= 2-detector components matching decoders require (surface code); False keeps
    whole faults for codes whose DEM is not graphlike (BB / qLDPC -- pair with
    bposd_window_decoder, since matching does not apply).

    `detector_rounds` maps global detector id -> 1-based round, for circuits whose
    detectors carry no time coordinates (e.g. QUITS-built BB circuits, where
    round = id // checks_per_round + 1). Default reads stim coordinates (t + 1)."""
    import numpy as np
    dem = circuit.detector_error_model(decompose_errors=decompose_errors)
    det_sets, obs_sets, priors = detector_error_model_to_faults(dem)
    n_obs = num_observables if num_observables is not None else circuit.num_observables
    if detector_rounds is not None:
        round_of = dict(detector_rounds)
    else:
        coords = circuit.get_detector_coordinates()
        coordless = sum(1 for c in coords.values() if not c)
        if coordless:
            raise ValueError(
                f"{coordless} detectors carry no coordinates; pass detector_rounds "
                "(global detector id -> 1-based round) explicitly")
        round_of = {det: int(c[-1]) + 1 for det, c in coords.items()}
    fault_rounds = [tuple(round_of[d] for d in dets) for dets in det_sets]

    models: list = []
    committed_elsewhere: set = set()               # fault indices owned by past windows
    last = len(plan) - 1
    for k, (commit_lo, commit_hi, buffer_hi) in enumerate(plan):
        # rows: this window's detectors (the last window keeps everything to the end)
        if k == last:
            rows = sorted(d for d, r in round_of.items() if r >= commit_lo)
        else:
            rows = sorted(d for d, r in round_of.items()
                          if commit_lo <= r <= buffer_hi)
        row_index = {d: i for i, d in enumerate(rows)}
        # columns: faults touching the rows, not committed by an earlier window
        cols = [f for f in range(len(det_sets))
                if f not in committed_elsewhere
                and any(d in row_index for d in det_sets[f])]
        check = np.zeros((len(rows), len(cols)), dtype=np.uint8)
        obs = np.zeros((n_obs, len(cols)), dtype=np.uint8)
        owned = np.zeros(len(cols), dtype=bool)
        future_flips: dict = {}
        for j, f in enumerate(cols):
            for d in det_sets[f]:
                if d in row_index:
                    check[row_index[d], j] = 1
            for o in obs_sets[f]:
                obs[o, j] = 1
            # ownership: the fault touches this window's commit region (any detector
            # at-or-before commit_hi -- earlier windows already took theirs), or this
            # is the last window (everything remaining must be decided)
            if k == last or any(r <= commit_hi for r in fault_rounds[f]):
                owned[j] = True
                committed_elsewhere.add(f)
                beyond = tuple(d for d in det_sets[f] if round_of[d] > commit_hi)
                if beyond and k != last:
                    future_flips[j] = beyond
        models.append(WindowErrorModel(
            detector_ids=tuple(rows), commit_hi=commit_hi,
            check=check, priors=np.array([priors[f] for f in cols]),
            obs=obs, owned=owned, future_flips=future_flips))
    return models


def decode_windowed(window_models: list, detection_events, decode_window) -> "object":
    """The committed-window decoding pass over one shot (the offline reference; the
    cluster performs the same steps event-by-event at runtime).

    For each window in order: take its detectors' bits, XOR in the artificial defects
    handed forward by earlier commits, decode, keep only the OWNED faults, accumulate
    their observable flips, and hand THEIR beyond-commit flips forward. Returns the
    predicted observable flips (XOR over all windows -- the convention the cluster's
    op_results already uses)."""
    import numpy as np
    pending: set = set()                           # artificial defects, by global det id
    total = np.zeros(window_models[0].obs.shape[0], dtype=np.uint8)
    for model in window_models:
        syndrome = detection_events[list(model.detector_ids)].astype(np.uint8).copy()
        for i, det in enumerate(model.detector_ids):
            if det in pending:
                syndrome[i] ^= 1
                pending.discard(det)
        selected = np.asarray(decode_window(model, syndrome), dtype=np.uint8)
        committed = selected.astype(bool) & model.owned
        total ^= (model.obs @ committed.astype(np.uint8)) % 2
        for col in np.nonzero(committed)[0]:
            for det in model.future_flips.get(int(col), ()):
                pending.symmetric_difference_update({det})   # defects XOR (mod 2)
    if pending:
        raise RuntimeError(f"artificial defects were never consumed: {sorted(pending)}"
                           " -- the plan does not cover the full detector stream")
    return total


def matching_window_decoder():
    """A PyMatching inner decoder for decode_windowed, caching one Matching per
    WindowErrorModel (the matrices are shot-independent). Boundary edges arise from
    single-detector columns; weights are the standard log((1-p)/p)."""
    import numpy as np
    import pymatching
    cache: dict = {}

    def decode(model: WindowErrorModel, syndrome):
        m = cache.get(id(model))
        if m is None:
            weights = np.log((1 - model.priors) / model.priors)
            m = pymatching.Matching.from_check_matrix(model.check, weights=weights)
            cache[id(model)] = m
        return m.decode(syndrome)

    return decode


def bposd_window_decoder(max_iter: int = 2, osd_order: int = 0,
                         bp_method: str = "product_sum", schedule: str = "serial",
                         osd_method: str = "osd_cs"):
    """A BP-OSD inner decoder for decode_windowed -- BB / qLDPC windows, whose faults
    may flip > 2 detectors (build the models with decompose_errors=False; matching
    does not apply). Defaults follow QUITS's sliding_window_bposd_* functions.
    Caches one ldpc.BpOsdDecoder per WindowErrorModel (matrices are shot-independent;
    only the syndrome changes per shot)."""
    cache: dict = {}

    def decode(model: WindowErrorModel, syndrome):
        d = cache.get(id(model))
        if d is None:
            from ldpc import BpOsdDecoder
            from scipy.sparse import csr_matrix
            d = BpOsdDecoder(csr_matrix(model.check),
                             error_channel=list(model.priors),
                             max_iter=max_iter, bp_method=bp_method,
                             schedule=schedule, osd_method=osd_method,
                             osd_order=osd_order)
            cache[id(model)] = d
        return d.decode(syndrome)

    return decode
