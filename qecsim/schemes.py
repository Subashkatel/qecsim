from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .message import Window, Operation
    from .protocols import CodeModel, LayoutModel

# ===============================================================================
# SCHEMES
# This is the default decoding scheme. It answers two questions:
# 1. how an operation's syndrome rounds are grouped into commit and buffer windows
# 2. when a window has accumulated enough syndrome rounds to be decoded safely.
# It is a pure POLICY -- no engine, no clock -- so it is a clean swap point for
# other windowing schemes: adaptive windowing (ADaPT, arXiv:2605.01149), parallel
# A/B-layer windowing (arXiv:2511.10633 Sec II.4), speculative windowing, double-
# window decoder switching (arXiv:2510.25222 Sec III.3), and so on.
# ===============================================================================

class SlidingWindowScheme:
    """The standard SEQUENTIAL (forward) sliding-window decoder: each operation is chopped
    into windows that commit C rounds behind a B-round look-ahead buffer, where C and B come
    from the code (both default to the code distance d). If the buffer spills beyond the end
    of the operation's own rounds, the overflow comes from the successor operation's early
    rounds (or idle memory rounds). Once a window commits, error strings crossing into its
    buffer become "artificial defects" handed to the NEXT window as its boundary -- so the
    windows of one operation form a serial dependency chain.

    Grounding. This is the (W, C)-sliding window of the literature: window size W = C + B
    with the commit-then-carry-forward artificial-defect rule (ADaPT, arXiv:2605.01149
    Sec II-C; used on FPGA hardware for the gross code with W = d and runtime-configurable C,
    arXiv:2510.21600 Sec 4.1). It is NOT the parallel windowing of arXiv:2511.10633 Sec II.4,
    where 3d-round windows have buffer/commit/buffer sub-regions and alternate in two layers
    (independent layer-A windows decode concurrently, then layer-B windows consume their
    boundaries; memory reaction time gamma_mem = 6d*tau_d(d^2) + t_com, Eq. 13). The serial
    chain here is throughput-limited by one window's decode per C rounds; the parallel scheme
    exists precisely to break that chain -- see ParallelWindowScheme below, which implements
    it through the scheme's own wire_deps hook with no planner change."""

    def plan_windows(self, op_id: int, n_rounds: int, code: CodeModel) -> list[tuple[int, int, int]]:
        """Lay out the windows for an operation: commit C rounds behind a B-round look-ahead buffer."""
        import math
        C,B,R = code.commit_rounds(), code.buffer_rounds(), n_rounds
        nwin = max(1, math.ceil(R / C))
        plan = []
        for k in range(nwin):
            commit_lo = k * C + 1
            commit_hi = min((k + 1) * C, R)
            buffer_hi = commit_hi + B
            plan.append((commit_lo, commit_hi, buffer_hi))
        return plan
    
    def data_complete(self, window: "Window", rounds_arrived: int, successor_rounds: int,
                      memory_rounds: int, n_rounds: int, has_successor: bool,
                      op: "Operation" = None, layout: "LayoutModel" = None) -> bool:
        """Return True once the commit+buffer rounds have arrived (including spillover from successor or memory rounds if needed).
        This default is a purely temporal rule; op/layout are available for more complex schemes (ignored here)."""
        in_op_need = min(window.buffer_hi, n_rounds)       # commit + in-op buffer rounds
        if rounds_arrived < in_op_need:
            return False
        overflow = window.buffer_hi - n_rounds  # buffer rounds beyond the end of the operation
        if overflow > 0:
            if not has_successor:
                return True # no successor to provide overflow rounds, so just go with what we have
            return (successor_rounds >= overflow) or (memory_rounds >= overflow) # successor or memory rounds can provide the overflow
        return True


class NaiveOnlineScheme(SlidingWindowScheme):
    """The NAIVE online decoding of arXiv:2510.25222 Sec III.C (Fig 9): all of an
    operation's syndrome data is collected first and then decoded COLLECTIVELY, as one
    batch -- no sliding, no look-ahead buffer. This is the baseline the windowing
    schemes exist to beat: decoding cannot even start until the last round has arrived,
    so the reaction wait carries the full batch decode, and a too-slow decoder shows
    the paper's backlog growth at its starkest (their Fig 10/11 use exactly this
    scheme). One window per operation; cross-op dependencies still apply (the planner's
    DAG wiring is scheme-independent).

    Do NOT expect it to lose on a single operation's reaction time: one batch has no
    serial window chain, no t_dd boundary hops, and no buffer-spillover wait, so it can
    beat the sliding scheme there -- the paper itself prefers batch-without-buffer when
    affordable (Sec III.C), and d/d sliding windows violate its Eq. 7 once tau_dec >
    tau_gen/2. The naive scheme's real cost is across a STREAM: decode never overlaps
    data collection, which is what drives the Fig 10/11 backlog growth.

    Inherits data_complete: with buffer_hi = n_rounds there is no overflow, so the
    window is ready exactly when all its own rounds have arrived.

    `batches_idle_rounds_into_next_op`: under this scheme a batch is the whole
    feedback-to-feedback SEGMENT -- the rounds a patch idled before the gate plus the
    gate's own rounds (the r_i of Eq. 5; Terhal's backlog argument: the record
    generated while waiting "needs to have been processed" before the next feedback).
    The cluster reads this flag in prepend_idle_rounds; continuously-windowed schemes
    leave it False and decode idle stretches concurrently instead (see
    docs/DESIGN-idle-stream-windows.md for why merging is exactly what makes this
    scheme reproduce Eq. 5 and concurrent windows would not)."""

    batches_idle_rounds_into_next_op = True

    def plan_windows(self, op_id: int, n_rounds: int, code: CodeModel) -> list[tuple[int, int, int]]:
        """One batch window: commit every round, look ahead none."""
        return [(1, n_rounds, n_rounds)]


class ParallelWindowScheme(SlidingWindowScheme):
    """The PARALLEL (two-layer) windowing of arXiv:2511.10633 Sec II.4. Windows have "3d
    temporal size" with "three d-sized sub-regions: a buffer region, a commit region, and
    another buffer region"; layer-A windows "are separated by a gap of d rounds"; each
    layer-B window covers "the two buffer regions and the gap region between two commit
    regions from layer A". Layer-A windows have NO dependencies on each other, so "all
    tasks in a single layer can, in principle, be decoded in parallel" -- with enough
    decoder units the decode latency is two window decodes, gamma_mem = 6d*tau_d(d^2) +
    t_com (Eq. 13), instead of the sequential scheme's one-decode-per-commit-stride chain.
    After a layer-A decode, the error strings crossing into its buffers become artificial
    defects at the buffer boundaries -- the t_dd boundary message each B window waits for.

    Layout, generically (C = commit_rounds, B = buffer_rounds, gap = C; the paper uses
    C = B = gap = d). Period S = 2C + 2B per A window:
      A_k (k >= 0): commit [1 + kS, kS + C], leading buffer B (none for A_0 -- the stream
                    has no rounds before round 1), trailing buffer B.
      B_k: commits everything between A_k's and A_{k+1}'s commit regions (trailing buffer
           + gap + leading buffer = 2B + C rounds); its lookahead on both sides is the A
           windows' data, enforced by its dependencies rather than extra rounds.
      tail: if rounds remain after the last A's commit, one final window commits them with
            a B-round lookahead (spillover from the successor, as in the sequential
            scheme). The stream-end handling is this implementation's choice; the paper
            describes the steady state only.

    Windows are emitted interleaved in commit order [A_0, B_0, A_1, B_1, ...], so EVEN
    indices are layer A and ODD indices are layer B / the tail -- which wire_deps uses.

    Inherits data_complete (a window is decodable once its own rounds arrived, with
    successor/memory spillover for lookahead past the operation's end)."""

    def plan_windows(self, op_id: int, n_rounds: int, code: CodeModel) -> list[tuple[int, int, int, int]]:
        """Lay out interleaved A/B windows: A commits every 2C+2B rounds, B commits the
        rounds in between, a tail window commits any stream-end remainder."""
        C, B, R = code.commit_rounds(), code.buffer_rounds(), n_rounds
        S = 2 * C + 2 * B
        a_windows = []                                     # (buffer_lo, commit_lo, commit_hi, buffer_hi)
        k = 0
        while 1 + k * S <= R:
            commit_lo = 1 + k * S
            commit_hi = min(commit_lo + C - 1, R)
            buffer_lo = max(1, commit_lo - B)              # A_0 has no leading rounds
            a_windows.append((buffer_lo, commit_lo, commit_hi, commit_hi + B))
            k += 1
        plan = []
        for i, a in enumerate(a_windows):
            plan.append(a)
            if i + 1 < len(a_windows):                     # B window between A_i and A_{i+1}
                lo, hi = a[2] + 1, a_windows[i + 1][1] - 1
                plan.append((lo, lo, hi, hi))              # pure commit; lookahead = A data
            elif a[2] < R:                                 # tail: commit the remainder
                lo = a[2] + 1
                plan.append((lo, lo, R, R + B))            # B-round lookahead (spillover)
        return plan

    def wire_deps(self, windows: list) -> None:
        """Layer-B windows (odd indices) depend on their neighbouring layer-A windows
        (boundary artificial defects from both sides); layer-A windows are independent."""
        for k in range(1, len(windows), 2):
            w = windows[k]
            w.deps.append((w.op_id, k - 1))                # A on the left
            if k + 1 < len(windows):
                w.deps.append((w.op_id, k + 1))            # A on the right (absent for tail)


# TODO: currently just a stub -- documents how decoder switching's windowing works; see the
# docstring recipe to finish it (plan_windows + the cluster escalation branch + the resume rule),
# then add acceptance tests for the paper's Eq. 15 constraint and Theorem 1 stability boundary.
class DoubleWindowScheme:
    """STUB. The double-window scheme of decoder switching (arXiv:2510.25222 Sec III.3).

    The runtime protocol to implement (verified against the paper):
      - A weak decoder processes ordinary (r_com, r_buf) sliding windows and returns a
        SOFT OUTPUT g with each result (DecodeResult.soft_output; Sec II.2-II.3 --
        complementary gap or cluster gap).
      - When g < g_th for some window, "the syndrome data of r_strong rounds, which
        includes the region with the small soft output", is assigned to the strong
        decoder, with r_strong = r_com + 2*r_buf, "after the boundary conditions at both
        ends have been determined by the weak decoder".
      - The weak decoder "resumes its process after allocating a data region of r_strong
        rounds to the strong decoder, and after r_com + r_buf rounds are subsequently
        stored".
      - No backlog (Theorem 1) iff: tau_weak < tau_gen; tau_strong <=
        gamma_switch^-1 * (d/r_strong) * tau_gen; r_com >= ceil(tau_weak/(tau_gen -
        tau_weak) * r_buf).

    How to finish it on the existing seams:
      1. plan_windows: like SlidingWindowScheme with (r_com, r_buf) from the code.
      2. The cluster's _on_decode_done branch: on result.soft_output < g_th, hold the
         window's commit, re-enqueue a DecodeJob covering r_strong rounds with
         attempt=1, hint="strong" (the DecoderRouter sends it to the strong decoder; the
         handoff is a weak->strong shipment, so charge the cluster's links.ws channel).
      3. data_complete of the NEXT window: require r_com + r_buf rounds beyond the
         escalated region (the resume rule above).
    Until then, use SwitchingDecoder (decoders.py) for timing-level switching studies --
    it models the same latency mix without the window interaction."""
    def plan_windows(self, op_id: int, n_rounds: int, code: CodeModel) -> list:
        """(stub) Would lay out (r_com, r_buf) windows; see the class docstring."""
        raise NotImplementedError("see DoubleWindowScheme docstring for the recipe; "
                                  "use SwitchingDecoder for timing-level studies meanwhile")
