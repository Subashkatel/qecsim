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