import heapq
import itertools
from dataclasses import dataclass, field
from typing import Callable, Optional
from .config import fmt

#===============================================================================
# THE DES ENGINE (generic event-driven simulator)
#===============================================================================

@dataclass(order=True)
class Event:
    """One scheduled thing to do at a future time. ordered by time, priority, insertion order so ties break deterministically."""
    time: int
    priority: int
    seq: int 
    action: Callable[[], None] = field(compare=False)
    label: str = field(compare=False, default="")

class Engine:
    """A minimal discrete event simulator a clock and a priority queue of events."""
    def __init__(self, verbose: bool = True):
        """ Create an empty simulator with the clock at 0 and an empty event queue."""
        self.now : int = 0
        self._q : list[Event] = []
        self._seq = itertools.count()
        self.verbose = verbose
        self.log_lines: list[str] = []
        self.metrics: list = []
        self.log_sink = None 

    def schedule(self, delay: int, action: Callable[[], None], label: str = "", priority: int = 0) -> None:
        """Push a future event onto the priority queue at (now + delay ticks) with some label and priority."""
        if delay < 0:
            raise ValueError(f"Cannot schedule an event in the past delay={delay} (now={self.now})")
        ev = Event(self.now + delay, priority, next(self._seq), action, label)
        heapq.heappush(self._q, ev) # put the event on the priority queue
        
    def log(self, who: str, msg: str) -> None:
        """ Log one timestamped line (print it if verbose otherwise send it to the log sink)."""
        line = f"[{fmt(self.now)}] {who}: {msg}"
        self.log_lines.append(line)
        if self.log_sink is not None:
            self.log_sink(line)
        if self.verbose:
            print(line)

    def add_metric(self, metric):
        """Add a metric to be observed at every event.(We will use later to track differnt performance metircs of what we are simulating)"""
        self.metrics.append(metric)
        return metric
    
    def metric_results(self) -> dict:
        """Get the results of all the metrics as a dictionary, keyed by the metric name."""
        return {metric.name: metric.result() for metric in self.metrics}

    def run(self, until: Optional[int] = None) -> None:
        """Run the simulation until there are no more events or until the time reaches 'until'."""
        while self._q:
            # check if the next event is in the future 
            if until is not None and self._q[0].time > until:
                break
            ev = heapq.heappop(self._q)
            if ev.time < self.now:
                raise ValueError(f"Event scheduled in the past: {ev} (now={self.now})")
            self.now = ev.time
            ev.action()
            for m in self.metrics:
                m.observe(self)

