from __future__ import annotations
 
from typing import TYPE_CHECKING
 
if TYPE_CHECKING:                      
    from .engine import Engine
# =====================================================================================
# METRICS
# This module defines the metrics that are pluggable read only observers of the simulation.
# For example, we can have a latency metric that measures the latency of the system 
# without affecting the rest of the system.
# =====================================================================================

class DecoderUtilization:
    """CONCRETE EXAMPLE. Time-weighted fraction of decoder units that were busy (0..1). Low
    utilization means decoders sat idle (over-provisioned or data-starved); near 1 means they
    are the bottleneck. Integrates the busy level held over each inter-event interval."""
    name = "decoder_utilization"
 
    def __init__(self, cluster):
        """Start the busy-time accumulator."""
        self.cluster = cluster
        self._t = 0
        self._busy_area = 0.0
        self._last_busy = 0
 
    def observe(self, engine: "Engine") -> None:
        """Add the busy level held since the last event, then read the current busy level."""
        self._busy_area += self._last_busy * (engine.now - self._t)
        self._t = engine.now
        self._last_busy = self.cluster.num_units - self.cluster.free_units
 
    def result(self) -> float:
        """Fraction of decoder-unit-time that was busy (0..1)."""
        return self._busy_area / (self.cluster.num_units * self._t) if self._t else 0.0

class ReadyQueueStats:
    """CONCRETE EXAMPLE. Peak and time-average length of the decoder ready queue how much
    decode work was waiting. Same time-weighted integration pattern as DecoderUtilization."""
    name = "ready_queue"
 
    def __init__(self, cluster):
        """Start the queue-length accumulator."""
        self.cluster = cluster
        self._t = 0
        self._area = 0.0
        self._last_len = 0
        self.peak = 0
 
    def observe(self, engine: "Engine") -> None:
        """Accumulate time-weighted queue length and track the peak."""
        self._area += self._last_len * (engine.now - self._t)
        self._t = engine.now
        self._last_len = len(self.cluster.ready)
        self.peak = max(self.peak, self._last_len)
 
    def result(self) -> dict:
        """Peak and time-average ready-queue length."""
        return {"peak": self.peak, "time_avg": (self._area / self._t if self._t else 0.0)}


class WindowLatencyBreakdown:
    """Per-window latency decomposition -- THE instrument for comparing windowing
    strategies (sequential vs parallel A/B vs adaptive; cf. ADaPT arXiv:2605.01149's
    window-size/cost tradeoffs, and the backlog signature of arXiv:2511.10633):

        BUFFER-FILL (first round -> data complete)   how long rounds buffer before usable
        DEP-BLOCK   (data complete -> queued)        waiting on predecessor boundaries
        QUEUE-WAIT  (queued -> dispatched)           waiting for a free decoder unit
        SERVICE     (dispatched -> done)             the decode itself

    The cluster stamps Window timestamps at events it already handles, so observe() is a
    no-op and registering this metric never changes the trace or the timing. result()
    gives per-stage mean/max; rows() gives one record per window for plotting/CSV.
    (Delivery to the orchestrator after the last window is the constant t_do hop and is
    not per-window data.)"""
    name = "window_latency"

    def __init__(self, cluster):
        """Hold the cluster whose windows carry the timestamps."""
        self.cluster = cluster

    def observe(self, engine: "Engine") -> None:
        """Nothing to sample (event-driven; the cluster stamps the windows)."""
        pass

    def rows(self) -> list:
        """One record per fully-decoded window: op, window index, and the four stages."""
        out = []
        for (op_id, k), w in sorted(self.cluster.windows.items()):
            stamps = (w.t_first_round, w.t_data_complete, w.t_queued, w.t_dispatch, w.t_done)
            if any(s is None for s in stamps):
                continue                       # window never (fully) decoded
            out.append({"op": op_id, "window": k,
                        "buffer_fill": w.t_data_complete - w.t_first_round,
                        "dep_block": w.t_queued - w.t_data_complete,
                        "queue_wait": w.t_dispatch - w.t_queued,
                        "service": w.t_done - w.t_dispatch,
                        "total": w.t_done - w.t_first_round})
        return out

    def result(self) -> dict:
        """Per-stage {mean, max, n} in ticks across all decoded windows."""
        rows = self.rows()
        stages = ("buffer_fill", "dep_block", "queue_wait", "service", "total")
        if not rows:
            return {s: {"mean": 0.0, "max": 0, "n": 0} for s in stages}
        return {s: {"mean": sum(r[s] for r in rows) / len(rows),
                    "max": max(r[s] for r in rows), "n": len(rows)} for s in stages}


class BacklogTrajectory:
    """Per-gate backlog -- the r_i of the decoder-switching paper (arXiv:2510.25222
    Sec III.C, Fig 9). For every gated operation it records the REACTION WAIT (from the
    gating operation's last physical round to the correction's return) and that wait
    expressed as accumulated syndrome rounds plus the gate's own rounds: the rounds the
    decoder must absorb between consecutive feedback decisions. A decoder that cannot
    keep up shows r_i GROWING with the gate index; a stable one converges.

    Reads timestamps the chip stamps at events it already handles (body_done_time,
    gate_release_time), so registering it never changes the trace or the timing --
    the same pattern as WindowLatencyBreakdown."""
    name = "backlog_trajectory"

    def __init__(self, chip):
        """Hold the chip whose timestamps we read."""
        self.chip = chip

    def observe(self, engine: "Engine") -> None:
        """Nothing to sample (event-driven; the chip stamps the timestamps)."""
        pass

    def rows(self) -> list:
        """One record per released gate, in release order: the reaction wait (ticks)
        and the backlog in rounds (wait / round time + the gate's own rounds)."""
        chip = self.chip
        out = []
        for op_id, t_release in sorted(chip.gate_release_time.items(),
                                       key=lambda kv: kv[1]):
            op = chip.ops[op_id]
            t_gate_open = chip.body_done_time.get(op.gated_by)
            if t_gate_open is None:
                continue                       # released before its gating op ran (never normal)
            wait = t_release - t_gate_open
            # rounds are counted in the OP'S OWN cadence (a per-code round_us override
            # changes how many rounds fit in the wait), not the chip's global one
            out.append({"op": op_id, "name": op.name, "released_at": t_release,
                        "wait": wait,
                        "backlog_rounds": wait / chip._round_ticks_for(op)
                                          + chip.cluster.rounds_for(op)})
        return out

    def result(self) -> dict:
        """Summary over all released gates: count, mean/max wait, mean/max backlog."""
        rows = self.rows()
        if not rows:
            return {"n": 0, "mean_wait": 0.0, "max_wait": 0,
                    "mean_backlog_rounds": 0.0, "max_backlog_rounds": 0.0}
        waits = [r["wait"] for r in rows]
        backlog = [r["backlog_rounds"] for r in rows]
        return {"n": len(rows),
                "mean_wait": sum(waits) / len(waits), "max_wait": max(waits),
                "mean_backlog_rounds": sum(backlog) / len(backlog),
                "max_backlog_rounds": max(backlog)}


class MagicStateLatency:
    """Production-side magic-state latency, from the factory's per-state StateTrace
    records (DistillationFactory): how long a state takes under a given production
    strategy x decoder strategy. Stages:

        distill     (distill start -> physical done)   the 15-to-1 rounds themselves
        corr_decode (physical done -> last corr done)  decoder-cluster coupling: queue
                                                       wait + the parallel correction
                                                       decodes (grows when the cluster is
                                                       contended -- arXiv:2511.10633's
                                                       reaction-time/factory coupling)
        deliver     (corr done -> delivered)           return trip + buffer idle

    Complements MagicStateStall (consumer-side wait). MultiLevelDistillationFactory's
    states are fungible buffer counts -- use its per-level produced/failures counters
    instead of this metric."""
    name = "magic_state_latency"

    def __init__(self, factory):
        """Hold the factory whose traces we summarize."""
        self.factory = factory

    def observe(self, engine: "Engine") -> None:
        """Nothing to sample (the factory stamps each state's StateTrace)."""
        pass

    def result(self) -> dict:
        """Per-stage {mean, max, n} in ticks across all delivered states."""
        traces = [t for t in getattr(self.factory, "traces", [])
                  if t.t_delivered is not None and t.t_corr_done is not None]
        stages = ("distill", "corr_decode", "deliver", "total")
        if not traces:
            return {s: {"mean": 0.0, "max": 0, "n": 0} for s in stages}
        vals = {"distill": [t.t_phys_done - t.t_distill_start for t in traces],
                "corr_decode": [t.t_corr_done - t.t_phys_done for t in traces],
                "deliver": [t.t_delivered - t.t_corr_done for t in traces],
                "total": [t.t_delivered - t.t_distill_start for t in traces]}
        return {s: {"mean": sum(v) / len(v), "max": max(v), "n": len(v)}
                for s, v in vals.items()}


# ---- STUBS: templates to copy. Each documents how to finish it; not registered by default. ----
# TODO: currently just a stub -- result() raises NotImplementedError; see docstring to finish it.
class DecodeLatencyHistogram:
    """STUB. Distribution of per-window queue-wait times. This data is event-driven, not
    sampled, so observe() stays empty; instead push a sample (engine.now - job.ready_time) from
    the cluster's _try_dispatch when a window is popped, into self.samples here. Then result()
    returns the histogram / percentiles."""
    name = "decode_latency_histogram"
    def __init__(self):
        """Start an empty sample list."""
        self.samples = []
    def observe(self, engine: "Engine") -> None:
        """Nothing to sample (event-driven; push samples at dispatch -- see docstring)."""
        pass
    def result(self):
        """(stub) Would return the wait-time distribution."""
        raise NotImplementedError("collect (now - job.ready_time) at dispatch; see docstring")
 
 
# TODO: currently just a stub -- a one-liner to finish; see docstring.
class MagicStateStall:
    """STUB. Total time operations waited on magic-state supply. Trivial to finish: hold the
    factory and return getattr(self.factory, 'total_stall', 0) from result()."""
    name = "magic_state_stall"
    def __init__(self, factory):
        """Hold the factory so we can read its stall total."""
        self.factory = factory
    def observe(self, engine: "Engine") -> None:
        """Nothing to sample."""
        pass
    def result(self):
        """(stub) Would return the factory's total supply-stall time."""
        raise NotImplementedError("return self.factory.total_stall (a one-liner); see docstring")






# TODO: currently just a stub -- needs a real error model + a failure-reporting decoder. 
class LogicalErrorRate:
    """STUB (domain). Logical error rate per operation -- requires a real error model on the
    DeviceModel + a decoding Decoder that reports failures, which this timing/structure DES does
    not model. Wire it once a StimDevice + real decoder are in place; observe() would count
    decode failures, result() would divide by operations."""
    name = "logical_error_rate"
    def __init__(self):
        """Start failure and operation counters."""
        self.failures = 0; self.ops = 0
    def observe(self, engine: "Engine") -> None:
        """Nothing to sample yet (needs a real error model)."""
        pass
    def result(self):
        """(stub) Would return failures / operations."""
        raise NotImplementedError("needs a real error model + failure-reporting decoder")