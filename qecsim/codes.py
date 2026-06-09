
from __future__ import annotations

from dataclasses import dataclass
# ================================================================================
# CODES
# The code model implementations everything code-specific the control
# and decoding simulation needs  so the rest of the engine stays code-agnostic
# we can swap in and out different codes without changing the rest of the system.
# ================================================================================

@dataclass(frozen=True)
class SurfaceCodeModel:
    """Rotated surface code, [[d^2, 1, d]] (arXiv:2411.03202 Sec 2.2). Conventions used
    here: one logical cycle = d syndrome rounds; decode windows commit d rounds behind a
    d-round look-ahead buffer (the W = 2d, C = d sliding window -- "conventionally W = d"
    per window-half in ADaPT arXiv:2605.01149 Sec II-C; arXiv:2511.10633 likewise builds
    its memory windows from d-sized commit/buffer sub-regions); the per-round decoding
    graph has ~d^2 nodes (arXiv:2511.10633 evaluates tau_d at N = d^2 per round)."""
    d: int = 3

    @property
    def name(self) -> str:
        """The code's human-readable name."""
        return f"rotated surface code (d={self.d})"

    @property
    def distance(self) -> int:
        """Code distance d (errors up to ~d/2 are corrected)."""
        return self.d

    def rounds_per_logical_cycle(self) -> int:
        """Syndrome rounds per logical cycle."""
        return self.d

    def rounds_per_op(self) -> int:
        """Temporal length of one operation = the logical cycle (d rounds) by default.
        Override to model a code that runs more or fewer rounds per operation."""
        return self.rounds_per_logical_cycle()

    def commit_rounds(self) -> int:
        """Rounds committed per decode window."""
        return self.d

    def buffer_rounds(self) -> int:
        """Look-ahead buffer rounds per window."""
        return self.d

    def spatial_nodes(self, num_patches: int) -> int:
        """Decoding-graph node count for this many patches (drives decode latency)."""
        npatch = max(1, num_patches)
        return npatch * self.d * self.d + (self.d if npatch > 1 else 0)

    def syndrome_bits_per_round(self, num_patches: int) -> int:
        """Syndrome bits measured per round."""
        return max(1, num_patches) * (self.d * self.d - 1)   # ~d^2-1 stabilizers per patch


# TODO: STUB -- structurally correct, but the node/round numbers are placeholders, not validated physics.
@dataclass(frozen=True)
class BBCodeModel:
    """Bivariate-bicycle "gross" code, [[144,12,12]]: 144 physical qubits encoding 12
    logical qubits at distance 12 (arXiv:2510.21600; also the dense-memory zone code of the
    heterogeneous architecture in arXiv:2411.03202 Sec 2.3). Decoded in W = d = 12-round
    sliding windows with runtime-configurable commit width on FPGA (arXiv:2510.21600
    Sec 4.1); pair with RelayBPDecoder for the matching latency model. The d-round
    circuit-level decoding matrix is 936 detectors x 8784 fault mechanisms
    (arXiv:2511.21660, verified)."""
    n: int = 144      # physical qubits
    k: int = 12       # logical qubits encoded in the block
    d: int = 12       # code distance
    num_checks: int = 132   # n - k independent stabilizer checks
    n_detectors: int = 936  # detectors in the d-round circuit decoding matrix (arXiv:2511.21660)
    n_faults: int = 8784    # fault-mechanism columns in the same matrix (arXiv:2511.21660)

    @property
    def name(self) -> str:
        """The code's human-readable name."""
        return f"bivariate-bicycle / gross code [[{self.n},{self.k},{self.d}]]"

    @property
    def distance(self) -> int:
        """Code distance d (errors up to ~d/2 are corrected)."""
        return self.d

    def rounds_per_logical_cycle(self) -> int:
        """Syndrome rounds per logical cycle."""
        return self.d

    def rounds_per_op(self) -> int:
        """Temporal length of one operation = the logical cycle (d rounds) by default.
        Override to model a code that runs more or fewer rounds per operation."""
        return self.rounds_per_logical_cycle()

    def commit_rounds(self) -> int:
        """Rounds committed per decode window."""
        return self.d

    def buffer_rounds(self) -> int:
        """Look-ahead buffer rounds per window."""
        return self.d

    def spatial_nodes(self, num_patches: int) -> int:
        # Per-round decoding-graph nodes ~ detectors-per-round of the gross-code circuit matrix.
        # Used for RESOURCE/RAM accounting and reporting; BB LATENCY comes from RelayBPDecoder,
        # NOT from this number (see class docstring).
        """Decoding-graph node count per round (detectors); does NOT set BP latency."""
        return max(1, num_patches) * (self.n_detectors // self.d)

    def syndrome_bits_per_round(self, num_patches: int) -> int:
        """Syndrome bits measured per round (~ checks per round)."""
        return max(1, num_patches) * (self.n_detectors // self.d)


# TODO: STUB -- parameterized placeholder; numbers are not validated physics.
@dataclass(frozen=True)
class ColorCodeModel:
    """Triangular color code """
    d: int = 3
    node_factor: float = 0.75   # ~3/4 d^2 data qubits for the triangular code

    @property
    def name(self) -> str:
        """The code's human-readable name."""
        return f"triangular color code (d={self.d}) (STUB)"

    @property
    def distance(self) -> int:
        """Code distance d (errors up to ~d/2 are corrected)."""
        return self.d

    def rounds_per_logical_cycle(self) -> int:
        """Syndrome rounds per logical cycle."""
        return self.d

    def rounds_per_op(self) -> int:
        """Temporal length of one operation = the logical cycle (d rounds) by default.
        Override to model a code that runs more or fewer rounds per operation."""
        return self.rounds_per_logical_cycle()

    def commit_rounds(self) -> int:
        """Rounds committed per decode window."""
        return self.d

    def buffer_rounds(self) -> int:
        """Look-ahead buffer rounds per window."""
        return self.d

    def spatial_nodes(self, num_patches: int) -> int:
        """Decoding-graph node count for this many patches (drives decode latency)."""
        return max(1, int(round(max(1, num_patches) * self.node_factor * self.d * self.d)))

    def syndrome_bits_per_round(self, num_patches: int) -> int:
        """Syndrome bits measured per round."""
        return self.spatial_nodes(num_patches)


# TODO: STUB -- parameterized placeholder; numbers are not validated physics.
@dataclass(frozen=True)
class ToricCodeModel:
    """Toric code """
    d: int = 3

    @property
    def name(self) -> str:
        """The code's human-readable name."""
        return f"toric code (d={self.d}) (STUB)"

    @property
    def distance(self) -> int:
        """Code distance d (errors up to ~d/2 are corrected)."""
        return self.d

    def rounds_per_logical_cycle(self) -> int:
        """Syndrome rounds per logical cycle."""
        return self.d

    def rounds_per_op(self) -> int:
        """Temporal length of one operation = the logical cycle (d rounds) by default.
        Override to model a code that runs more or fewer rounds per operation."""
        return self.rounds_per_logical_cycle()

    def commit_rounds(self) -> int:
        """Rounds committed per decode window."""
        return self.d

    def buffer_rounds(self) -> int:
        """Look-ahead buffer rounds per window."""
        return self.d

    def spatial_nodes(self, num_patches: int) -> int:
        """Decoding-graph node count for this many patches (drives decode latency)."""
        return max(1, num_patches) * 2 * self.d * self.d

    def syndrome_bits_per_round(self, num_patches: int) -> int:
        """Syndrome bits measured per round."""
        return max(1, num_patches) * 2 * self.d * self.d