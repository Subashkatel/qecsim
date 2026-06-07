from dataclasses import dataclass, field

#===============================================================================
# CONFIGURATION 
#===============================================================================

# Time Units : The simulator uses integers ticks (simiilar to classical gem5 simulator)
# 1 tick = 1 picosecond (global frequency)
TICKS_PER_US = 1_000_000

def us(microseconds : float) -> int:
    """Convert microseconds to ticks."""
    return int(round(microseconds * TICKS_PER_US))

def fmt(ticks : int ) -> str:
    """Format ticks as microseconds for readiability in logs."""
    return f"{ticks / TICKS_PER_US:7.3f} us"


# @dataclass(frozen=True)
# class SimConfig:
#     """All the changable parameters in one place so its easy to modify"""
#     rounds_us: float = 1.1 # one syndrome round = one parity check cycle 
#     rounds_per_op: int = 11 # number of rounds per logical operation (two_qubit op + bus)

#     t_qc_us: float = 0.15  # chip -> controller latency (microseconds)
#     t_cd_us: float = 2.0   # controller -> decoder cluster latency (microseconds)
#     t_dd_us: float = 0.5   # decoder -> decoder message passing latency (microseconds)
#     t_do_us: float = 1.0   # decoder -> orchestrator latency (microseconds)
#     t_oc_us: float = 4.0   # orchestrator -> controller latency (microseconds)
#     t_cq_us: float = 0.15  # controller -> chip latency (microseconds)
