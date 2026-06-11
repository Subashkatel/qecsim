"""Shared test helpers."""


def trace_time(lines, needle):
    """Timestamp (us, float) of the first trace line containing `needle`."""
    for line in lines:
        if needle in line:
            return float(line.split("us]")[0].lstrip("["))
    raise AssertionError(f"no trace line contains {needle!r}")
