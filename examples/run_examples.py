"""Runnable examples for qecsim.

Run all of them:
    python examples/run_examples.py

Run just one (by number), or a few:
    python examples/run_examples.py 2
    python examples/run_examples.py 1 3
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from qecsim import (build_and_run, three_cnot_circuit, cnot_plus_two_t_circuit,
                    independent_t_circuit, PresetLatencyDecoder, RelayBPDecoder,
                    DistillationFactory, DecoderUtilization, ReadyQueueStats, us,
                    ParallelWindowScheme, WindowLatencyBreakdown)
from qecsim.frontends.circuit import CircuitFrontend
from qecsim.message import Operation

D, RPO = 3, 11


def example1():
    """THREE CNOTs, 1 decoder unit."""
    build_and_run(three_cnot_circuit(), num_units=1, d=D, rounds_per_op=RPO,
                  decoder=PresetLatencyDecoder(1.0),
                  title="1) THREE CNOTs, 1 decoder unit")


def example2():
    """CNOT + two T gates: gating, magic states, conditional dispatch."""
    build_and_run(cnot_plus_two_t_circuit(), num_units=2, d=D, rounds_per_op=RPO,
                  title="2) CNOT + two T gates: gating, magic states, conditional dispatch")


def example3():
    """SIX T gates, undersized factory: large magic-state stall."""
    build_and_run(independent_t_circuit(6), num_units=8, d=D, rounds_per_op=RPO,
                  make_factory=lambda e, cl: DistillationFactory(
                      e, num_units=1, cycle_ticks=us(11 * RPO * 1.1), decode_service=cl,
                      corr_rounds=2 * D, n_corr=11, return_ticks=us(5.0), initial_store=0),
                  title="3) SIX T GATES, undersized factory: large magic-state stall")


def example4():
    """Relay-BP decoder + metrics."""
    r = build_and_run(three_cnot_circuit(), num_units=1, d=D, rounds_per_op=RPO,
                      decoder=RelayBPDecoder(),
                      make_metrics=lambda e, cl, ch, f: [DecoderUtilization(cl), ReadyQueueStats(cl)],
                      title="4) Relay-BP decoder + metrics")
    print("   metrics:", {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in r["metrics"].items()})


def example5():
    """Windowing study: sequential vs parallel A/B (arXiv:2511.10633 Sec II.4) on a
    memory stream, with the per-stage latency breakdown showing WHERE the time goes."""
    mem = CircuitFrontend([Operation(0, "M(q0)", (0,), clifford=True)]).build()
    for name, scheme in (("sequential", None), ("parallel A/B", ParallelWindowScheme())):
        r = build_and_run(mem, num_units=4, d=D, rounds_per_op=63,
                          decoder=PresetLatencyDecoder(10.0), scheme=scheme,
                          make_metrics=lambda e, cl, ch, f: [WindowLatencyBreakdown(cl)],
                          verbose=False,
                          title=f"5) WINDOWING STUDY -- {name}, 4 units, slow decoder")
        stages = r["metrics"]["window_latency"]
        print("   per-window stage means (us): " +
              ", ".join(f"{s}={stages[s]['mean'] / 1e6:.2f}"
                        for s in ("buffer_fill", "dep_block", "queue_wait", "service")))


# example number -> function. Add a new example by registering it here.
EXAMPLES = {1: example1, 2: example2, 3: example3, 4: example4, 5: example5}


def main(argv):
    """No arguments -> run every example. Otherwise run only the numbers given."""
    if not argv:
        for n in sorted(EXAMPLES):
            EXAMPLES[n]()
        return
    for arg in argv:
        if not arg.isdigit() or int(arg) not in EXAMPLES:
            raise SystemExit(f"no example {arg!r}; choose from {sorted(EXAMPLES)}")
    for arg in argv:
        EXAMPLES[int(arg)]()


if __name__ == "__main__":
    main(sys.argv[1:])
