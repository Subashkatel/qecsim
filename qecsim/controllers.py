from __future__ import annotations

from typing import Callable

from .config import us
from .engine import Engine
from .message import SyndromePayload, Decision
# ================================================================================
# CONTROLLER
# The controller is the classical unit attached to the QPU (arXiv:2511.10633 Sec III;
# one controller manages ~10^4 physical qubits there). It sits on BOTH sides of the
# reaction path: outbound it relays each round's syndromes chip -> controller (t_qc)
# -> decoder cluster (t_cd); inbound it relays conditional instructions orchestrator
# -> controller (t_oc) -> chip (t_cq). It also owns the two decoder-side link
# constants: decoder->decoder boundary exchange (t_dd) and decoder->orchestrator
# result delivery (t_do).
# ================================================================================

class ModularController:
    """The default controller implementation.

    The six link latencies default to Table 2 of arXiv:2511.10633:
        t_qc = 0.15 us (QPU -> controller, 1 bit/channel)
        t_cd = 2 us    (controller -> decoders, ~5000 bits)
        t_dd = 0.5 us  (decoder -> decoder boundary defects, ~100 bits; ~90 bits at d=30)
        t_do = 1 us    (decoders -> orchestrator, ~50000 bits)
        t_oc = 4 us    (orchestrator -> controller, ~20000 bits)
        t_cq = 0.15 us (controller -> QPU, 1 bit/channel)
    summing to the paper's end-to-end communication time t_com ~= 10 us.

    The controller is the paper's AGGREGATION point (arXiv:2511.10633 Sec III.1): it
    discriminates per-qubit readout into "1-2 bits per qubit" and forwards BATCHED
    packets to the decoders over t_cd. Accordingly, when a round arrives as several
    fragments (SyndromePayload.n_fragments > 1, e.g. per-patch payloads or staggered
    readout chunks), each fragment pays t_qc on its own, the controller buffers them,
    and the round's LAST fragment triggers ONE packet: t_pack (the packaging/
    serialization cost, default 0) + one t_cd, delivering the whole round to the
    decoder cluster atomically. Whole-round payloads (n_fragments <= 1, the default
    devices) skip the buffer entirely -- the original two-hop path, unchanged."""

    def __init__(self, engine: Engine, t_qc=us(0.15), t_cd=us(2.0), t_dd=us(0.5),
                 t_do=us(1.0), t_oc=us(4.0), t_cq=us(0.15), log_syndromes=True,
                 t_pack=0):
        """Store the six link latencies (constants from arXiv:2511.10633 Table 2) and
        the per-packet packaging cost t_pack (ticks)."""
        self.engine = engine
        self.t_qc = t_qc
        self.t_cd = t_cd
        self.t_dd = t_dd
        self.t_do = t_do
        self.t_oc = t_oc
        self.t_cq = t_cq
        self.t_pack = t_pack
        self.log_syndromes = log_syndromes
        self._pending: dict = {}      # (op_id, round_index) -> [(payload, deliver), ...]

    def relay_syndrome(self, payload: SyndromePayload,
                       deliver: Callable[[SyndromePayload], None]) -> None:
        """Send a syndrome chip->controller->decoders. A whole-round payload is forwarded
        after the two hop delays; a fragment of a multi-fragment round is buffered at the
        controller until the round is complete, then the round ships as ONE packet."""
        if payload.n_fragments == 1:
            def at_controller():
                """Second controller hop: deliver to the destination after the link delay."""
                if self.log_syndromes:
                    self.engine.log("Controller",
                                    f"received round {payload.round_index} of "
                                    f"op#{payload.operation_id} from chip (t_qc); "
                                    f"forwarding to decoder (t_cd)")
                self.engine.schedule(self.t_cd, lambda: deliver(payload),
                                     label="controller->decoder")
            self.engine.schedule(self.t_qc, at_controller, label="chip->controller")
            return

        def at_controller_fragment():
            # NOTE : For future usecase where if the qpu device doesn't send the full
            # rounds syndrome so the controller needs to buffer the fragments until 
            # the full round is received, then package and forward to the decoder. 
            # The current implementation of the qpu device sends the full round 
            # syndrome so this code is not used, but it is implemented for future usecase.
            """Buffer this fragment; on the round's last fragment, package and forward."""
            key = (payload.operation_id, payload.round_index)
            buf = self._pending.setdefault(key, [])
            buf.append((payload, deliver))
            if len(buf) < payload.n_fragments:
                if self.log_syndromes:
                    self.engine.log("Controller",
                                    f"buffered fragment {len(buf)}/{payload.n_fragments} "
                                    f"of round {payload.round_index} of "
                                    f"op#{payload.operation_id} (waiting for the rest)")
                return
            del self._pending[key]
            if self.log_syndromes:
                self.engine.log("Controller",
                                f"round {payload.round_index} of op#{payload.operation_id} "
                                f"complete ({payload.n_fragments} fragments); packaging "
                                f"and forwarding ONE packet to decoder (t_pack + t_cd)")
            self.engine.schedule(self.t_pack + self.t_cd,
                                 lambda b=tuple(buf): [d(p) for p, d in b],
                                 label="controller->decoder packet")
        self.engine.schedule(self.t_qc, at_controller_fragment, label="chip->controller")

    def relay_instruction(self, decision: "Decision",
                         deliver: Callable[["Decision"], None]) -> None:
        """Send a correction orchestrator->controller->chip, delivering after the hop delays."""
        def at_controller():
            """Second controller hop: deliver to the destination after the link delay."""
            self.engine.log("Controller",
                            f"received instruction for op#{decision.gadget_id} from "
                            f"orchestrator (t_oc); forwarding to chip (t_cq)")
            self.engine.schedule(self.t_cq, lambda: deliver(decision),
                                 label="controller->chip")
        self.engine.schedule(self.t_oc, at_controller, label="orchestrator->controller")

    # TODO(link model): these two constants are decoder-side links the controller is not
    # physically on; they live here so all six Table-2 numbers form one price list. When
    # the decoder-switching study activates, decide whether they -- plus a weak<->strong
    # handoff channel, possibly size-aware (latency = fixed + bits/bandwidth, using the
    # Table-2 payload sizes: ~100 bits boundary, ~5000 bits/round) -- should move to a
    # dedicated LinkModel seam consulted by both controller and cluster.
    def dec_to_dec_delay(self) -> int:
        """Decoder-to-decoder boundary-message latency (ticks)."""
        return self.t_dd                        # artificial-defect handoff between windows

    def dec_to_orch_delay(self) -> int:
        """Decoder-to-orchestrator message latency (ticks)."""
        return self.t_do                        # decoders -> orchestrator
