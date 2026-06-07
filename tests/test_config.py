from qecsim.config import TICKS_PER_US, us , fmt

def test_ticks_per_us():
    assert TICKS_PER_US == 1_000_000

def test_us():
    assert us(1.1) == 1_100_000
    assert us(2.0) == 2_000_000

def test_fmt():
    assert fmt(1_100_000) == "  1.100 us"
    assert fmt(2_000_000) == "  2.000 us"

def test_fmt_output():
    print(repr(fmt(1_100_000)))
