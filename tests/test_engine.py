#==================================================================
# TESTS FOR ENGINE
#==================================================================
import pytest
from qecsim.engine import Engine

def test_engine():
    eng = Engine(verbose=False)
    order = []
    eng.schedule(10, lambda: order.append(10), label="This is 10")
    eng.schedule(5, lambda: order.append(5), label="This is 5")
    eng.schedule(15, lambda: order.append(15), label="This is 15")
    eng.schedule(5, lambda: order.append(5.5), label="This is 5.5", priority=-1)
    eng.run()
    assert order == [5.5, 5, 10, 15]

def test_engine_log():
    eng = Engine(verbose=False)
    eng.schedule(10, lambda: eng.log("Test", "This is 10"), label="This is 10")
    eng.schedule(5, lambda: eng.log("Test", "This is 5"), label="This is 5")
    eng.schedule(15, lambda: eng.log("Test", "This is 15"), label="This is 15")
    eng.schedule(20, lambda: eng.log("Test", "This is 20"), label="This is 20")
    eng.run()
    assert len(eng.log_lines) == 4
    assert "This is 20" in eng.log_lines[3]
    assert "This is 5" in eng.log_lines[0]
    assert "This is 10" in eng.log_lines[1]
    assert "This is 15" in eng.log_lines[2]

def test_engine_schedule_past():
    eng = Engine(verbose=False)
    with pytest.raises(ValueError):
        eng.schedule(-10, lambda: None, label="This is in the past")

def test_event_within_event():
    """Tests one event that schedules another future event."""
    eng = Engine(verbose=False)
    log = []

    def tick(n):
        log.append((eng.now, n))
        if n < 4:
            eng.schedule(delay=10, action=lambda: tick(n+1), label=f"Tick {n+1}")

    eng.schedule(delay=0, action=lambda: tick(1), label="Tick 1")
    eng.run()
    assert log == [(0, 1), (10, 2), (20, 3), (30, 4)]

def test_event_schedules_multiple_events():
    """Tests one event that schedules multiple future events."""
    eng = Engine(verbose = False)
    log = []
    def child(name):
        log.append((eng.now, name))

    def parent():
        log.append((eng.now, "parent"))
        eng.schedule(delay=10, action=lambda: child("child1"), label="Child 1")
        eng.schedule(delay=20, action=lambda: child("child2"), label="Child 2")
    eng.schedule(delay=0, action=parent, label="Parent")
    eng.run()
    assert log == [(0, "parent"), (10, "child1"), (20, "child2")]



# TODO: test_engine_determinism
# TODO: test_engine_metrics
# TODO: test_engine_until
# TODO: test_engine_log_sink
# TODO: test_engine_empty_run
        
