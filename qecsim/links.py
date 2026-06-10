
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from .config import us

# ================================================================================
# LINKS
# The classical communication fabric: one price list for every hop in the stack.
# The six named links and their default latencies are Table 2 of arXiv:2511.10633
# (each is justified there by a payload size -- see the Link docstring); `ws` is
# the weak<->strong decoder escalation channel of arXiv:2510.25222 (its T_comm).
# By default every link is a FLAT constant (infinite bandwidth, no sharing), which
# reproduces the original behavior exactly. Two opt-ins per link:
#   - a bandwidth makes cost respond to message SIZE (the hardware-papers' regime:
#     syndrome traffic is a throughput problem -- arXiv:2512.09807, arXiv:2303.00054);
#   - serialize=True makes the link a shared bus that transmits one message at a
#     time, so contention EMERGES (e.g. one cryo feedthrough carrying every patch).
# ================================================================================

@dataclass
class Link:
    """One classical channel. The cost of sending one message is
        latency_ticks                                   (propagation + protocol)
      + bits / bandwidth_bits_per_us                    (serialization, if sized)
      + time until the bus is free                      (only if serialize=True)
    Defaults make it a flat constant. Typical payloads (arXiv:2511.10633 Table 2):
    ~1 bit/channel readout, ~5000 bits a batched round, ~100 bits a boundary-defect
    message, ~50000 bits a result."""
    latency_ticks: int
    bandwidth_bits_per_us: Optional[float] = None  # None = infinite (size ignored)
    serialize: bool = False                        # True = shared bus, queues messages
    next_free_tick: int = 0                        # shared-bus bookkeeping (internal)

    def __post_init__(self):
        if self.latency_ticks < 0:
            raise ValueError(f"latency_ticks must be >= 0 (got {self.latency_ticks})")
        if self.bandwidth_bits_per_us is not None and self.bandwidth_bits_per_us <= 0:
            raise ValueError(f"bandwidth_bits_per_us must be > 0 "
                             f"(got {self.bandwidth_bits_per_us})")

    def cost(self, bits: Optional[int] = None, now: Optional[int] = None) -> int:
        """Delay (ticks from now) until a message of `bits` is delivered. Pass `now`
        (engine.now) for a serialized link so it can queue behind the bus; flat links
        ignore both arguments. The bus is occupied for the serialization time only --
        propagation overlaps the next message (a pipelined wire)."""
        serialization = us(bits / self.bandwidth_bits_per_us) \
            if (bits is not None and self.bandwidth_bits_per_us is not None) else 0
        if self.serialize and now is not None:
            start = max(now, self.next_free_tick)
            self.next_free_tick = start + serialization
            return (start - now) + serialization + self.latency_ticks
        return serialization + self.latency_ticks


def _as_link(value: Union[int, Link]) -> Link:
    """Accept a plain latency (ticks) or a full Link object."""
    return value if isinstance(value, Link) else Link(int(value))


class LinkModel:
    """The named links of the architecture, one object shared by every component
    (the wiring threads a single instance to the controller and the cluster, so the
    fabric cannot disagree with itself):

        qc  chip -> controller            cd  controller -> decoder cluster
        dd  decoder -> decoder            do  decoders -> orchestrator
        oc  orchestrator -> controller    cq  controller -> chip
        ws  weak <-> strong decoder escalation (decoder switching's T_comm; defaults
            to the dd value until a study prices the FPGA->GPU hop differently)

    Each argument is a latency in ticks (flat link) or a Link object (size-aware /
    shared-bus)."""
    def __init__(self, qc: Union[int, Link] = us(0.15), cd: Union[int, Link] = us(2.0),
                 dd: Union[int, Link] = us(0.5), do: Union[int, Link] = us(1.0),
                 oc: Union[int, Link] = us(4.0), cq: Union[int, Link] = us(0.15),
                 ws: Union[int, Link] = us(0.5)):
        """Hold the seven links (defaults: arXiv:2511.10633 Table 2; ws = the dd value)."""
        self.qc = _as_link(qc)
        self.cd = _as_link(cd)
        self.dd = _as_link(dd)
        self.do = _as_link(do)
        self.oc = _as_link(oc)
        self.cq = _as_link(cq)
        self.ws = _as_link(ws)
