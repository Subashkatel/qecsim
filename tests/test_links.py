"""The communication fabric (links.py): flat constants by default (byte-identical to
the old controller-owned scalars), opt-in size-aware serialization, opt-in shared-bus
queueing, and the single-source threading through wiring."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from qecsim.config import us
from qecsim.links import Link, LinkModel
from qecsim.message import Operation
from qecsim.wiring import build_and_run


def test_default_link_model_matches_table_2():
    """The defaults are the six Table-2 constants (arXiv:2511.10633) plus ws = t_dd."""
    links = LinkModel()
    assert links.qc.cost() == us(0.15)
    assert links.cd.cost() == us(2.0)
    assert links.dd.cost() == us(0.5)
    assert links.do.cost() == us(1.0)
    assert links.oc.cost() == us(4.0)
    assert links.cq.cost() == us(0.15)
    assert links.ws.cost() == us(0.5)


def test_flat_link_ignores_message_size():
    """Without a bandwidth, a link is a flat constant -- bits change nothing."""
    flat = Link(us(2.0))
    assert flat.cost() == us(2.0)
    assert flat.cost(bits=5000) == us(2.0)


def test_bandwidth_link_adds_serialization_time():
    """With a bandwidth, cost = latency + bits/bandwidth (a 5000-bit round over
    2500 bits/us takes 2 us to serialize on top of the 2 us propagation)."""
    sized = Link(us(2.0), bandwidth_bits_per_us=2500.0)
    assert sized.cost(bits=5000) == us(2.0) + us(2.0)
    assert sized.cost() == us(2.0)            # no bits given -> latency only


def test_serialized_link_queues_messages():
    """A shared bus transmits one message at a time: the second same-instant message
    waits for the first one's serialization slot."""
    bus = Link(us(1.0), bandwidth_bits_per_us=1000.0, serialize=True)
    assert bus.cost(bits=1000, now=0) == us(2.0)   # serialize 1 us + propagate 1 us
    assert bus.cost(bits=1000, now=0) == us(3.0)   # waits 1 us for the bus first
    # a serialized link with no bandwidth occupies the bus for zero time -> no queueing
    free_bus = Link(us(1.0), serialize=True)
    assert free_bus.cost(now=0) == us(1.0)
    assert free_bus.cost(now=0) == us(1.0)


def test_links_validate_loudly():
    with pytest.raises(ValueError):
        Link(-1)
    with pytest.raises(ValueError):
        Link(us(1.0), bandwidth_bits_per_us=0.0)


def test_link_model_accepts_link_overrides():
    """A whole Link object may replace a plain latency (e.g. a size-aware cd link)."""
    links = LinkModel(cd=Link(us(1.0), bandwidth_bits_per_us=2500.0))
    assert links.cd.cost(bits=5000) == us(1.0) + us(2.0)
    assert links.qc.cost() == us(0.15)             # others keep their defaults


def test_wiring_threads_one_link_model_to_controller_and_cluster():
    """Single source of truth: the controller and the cluster share the SAME LinkModel
    object, so the fabric cannot disagree with itself."""
    ops = [Operation(0, "M(q0)", (0,), clifford=True)]
    r = build_and_run(ops, num_units=1, d=3, rounds_per_op=11, verbose=False)
    assert r["controller"].links is r["cluster"].links
