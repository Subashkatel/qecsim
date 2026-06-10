from __future__ import annotations
 
from dataclasses import dataclass
from typing import TYPE_CHECKING
 
if TYPE_CHECKING:                     
    from .engine import Engine
    from .protocols import CodeModel, Controller, Decoder

#===============================================================================
# CONFIGURATION 
#===============================================================================

# Time Units : The simulator uses integer ticks (similar to the classical gem5 simulator)
# 1 tick = 1 picosecond (global frequency)
TICKS_PER_US = 1_000_000

def us(microseconds : float) -> int:
    """Convert microseconds to ticks."""
    return int(round(microseconds * TICKS_PER_US))

def fmt(ticks : int ) -> str:
    """Format ticks as microseconds for readability in logs."""
    return f"{ticks / TICKS_PER_US:7.3f} us"


@dataclass(frozen=True)
class SimConfig:
    """All the changable parameters in one place so its easy to modify.

    Defaults are grounded in Khalid et al., arXiv:2511.10633: the six link latencies are
    Table 2; (decoder_alpha, decoder_beta) are the Table 3 monomial fit tau_d(N)=alpha*N^beta
    for the Collision Cluster decoder on FPGA. Syndrome-round time is platform-dependent
    (~1 us superconducting: 1 us/cycle in arXiv:2510.21600; ~0.5 us stabilization rounds in
    arXiv:2411.10406 Sec I.2.1)."""
    round_us: float = 1.1 # one syndrome round = one parity check cycle
    rounds_per_op: int = 11 # number of rounds per logical operation (two_qubit op + bus)
    num_units: int = 1             # decoder units in the cluster

    # link latencies: Table 2 of arXiv:2511.10633 (sum ~= t_com ~ 10 us)
    t_qc_us: float = 0.15  # chip -> controller latency (microseconds)
    t_cd_us: float = 2.0   # controller -> decoder cluster latency (microseconds)
    t_dd_us: float = 0.5   # decoder -> decoder message passing latency (microseconds)
    t_do_us: float = 1.0   # decoder -> orchestrator latency (microseconds)
    t_oc_us: float = 4.0   # orchestrator -> controller latency (microseconds)
    t_cq_us: float = 0.15  # controller -> chip latency (microseconds)
    t_pack_us: float = 0.0 # controller packaging cost per round packet (microseconds): the controller aggregates a round's fragments into one t_cd packet (arXiv:2511.10633 Sec III.1); this prices the serialization/compression step (0 = free, the paper folds it into t_cd)

    # Decoder speed model tau_d(N) = alpha * N^beta (arXiv:2511.10633 Eq. 12): time to decode
    # one round of a decoding graph with N nodes (N ~ d^2 for a distance-d patch).
    #   alpha = the hardware's raw speed, in seconds (smaller = faster decoder)
    #   beta  = how decode time grows with patch size (>1 = superlinear: doubling the
    #           graph more than doubles the decode time)
    # Defaults: the paper's Table 3 fit for the Collision Cluster decoder on FPGA. Other
    # Table 3 fits to swap in: ASIC (5.53e-11, 1.34), AlphaQubit (4.8e-6, 0.503),
    # PyMatching at p=0.1% (5.91e-9, 1.17).
    decoder_alpha: float = 2.85e-10
    decoder_beta: float = 1.2

    def __post_init__(self):
        """ This method is called after the dataclass is initialized so it can perform validation."""
        if self.round_us <= 0:
            raise ValueError(f"round_us must be > 0 (got {self.round_us})")
        if self.rounds_per_op < 1:
            raise ValueError(f"rounds_per_op must be >= 1 (got {self.rounds_per_op})")
        if self.num_units < 1:
            raise ValueError(f"num_units must be >= 1 (got {self.num_units})")
        for name in ("t_qc_us", "t_cd_us", "t_dd_us", "t_do_us", "t_oc_us", "t_cq_us",
                     "t_pack_us"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0 (got {getattr(self, name)})")
        if self.decoder_alpha < 0 or self.decoder_beta < 0:
            raise ValueError("decoder_alpha and decoder_beta must be >= 0")

    def make_controller(self, engine: "Engine") -> "Controller":
        """Build the modular controller from these link-latency knobs. Imported lazily so this
        module has no import-time dependency on controllers.py (which imports `us` from here)."""
        from .controllers import ModularController
        return ModularController(engine,
                                 t_qc=us(self.t_qc_us), t_cd=us(self.t_cd_us),
                                 t_dd=us(self.t_dd_us), t_do=us(self.t_do_us),
                                 t_oc=us(self.t_oc_us), t_cq=us(self.t_cq_us),
                                 t_pack=us(self.t_pack_us))

    def make_decoder(self, code: "CodeModel") -> "Decoder":
        """Build the default latency-model decoder for the given code (lazy import, as above)."""
        from .decoders import LatencyModelDecoder
        return LatencyModelDecoder(d=code.distance,
                                   alpha=self.decoder_alpha, beta=self.decoder_beta)
