from qecsim.message import Operation

def test_operation_needs_magic_state():
    assert Operation(0, "CNOT", (0, 1), clifford=True).needs_magic_state is False
    assert Operation(1, "T", (2,), clifford=False).needs_magic_state is True
