from __future__ import annotations

from typing import Callable

from config import us
from engine import Engine
from message import SyndromePayload, Decision
# ================================================================================
# CONTROLLER
# The controller the paper's "modular controller" (arXiv:2511.10633 Sec III).
# In the paper it is the classical unit attached to the QPU both side reaction path
# ================================================================================

class ModularController:
    """The default controller implementation"""

    def __init__(self, engine: Engine, t_qc=us(0.15), t_cd=us(2.0), t_dd=us(0.5),
                 t_do=us(1.0), t_oc=us(4.0), t_cq=us(0.15), log_syndromes=True):
        """Store the six link latencies (CONSTANTS from Table II)."""
        self.engine = engine
        self.t_qc = t_qc
        self.t_cd = t_cd
        self.t_dd = t_dd
        self.t_do = t_do
        self.t_oc = t_oc
        self.t_cq = t_cq
        self.log_syndromes = log_syndromes

    def relay_syndrome(self, payload: SyndromePayload,
                       deliver: Callable[[SyndromePayload], None]) -> None:
        """Send a syndrome chip->controller->decoders, delivering after the two hop delays."""
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

    def dec_to_dec_delay(self) -> int:
        """Decoder-to-decoder boundary-message latency (ticks)."""
        return self.t_dd                        # artificial-defect handoff between windows
    
    def dec_to_orch_delay(self) -> int:
        """Decoder-to-orchestrator message latency (ticks)."""
        return self.t_do                        # decoders -> orchestrator