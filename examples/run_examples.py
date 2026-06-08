 
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
 
from qecsim import (build_and_run, three_cnot_circuit, cnot_plus_two_t_circuit,
                    independent_t_circuit, PresetLatencyDecoder, RelayBPDecoder,
                    DistillationFactory, DecoderUtilization, ReadyQueueStats, us)
 
D, RPO = 3, 11
 
build_and_run(three_cnot_circuit(), num_units=1, d=D, rounds_per_op=RPO,
              decoder=PresetLatencyDecoder(1.0),
              title="1) THREE CNOTs, 1 decoder unit")
 
build_and_run(cnot_plus_two_t_circuit(), num_units=2, d=D, rounds_per_op=RPO,
              title="2) CNOT + two T gates: gating, magic states, conditional dispatch")
 
build_and_run(independent_t_circuit(6), num_units=8, d=D, rounds_per_op=RPO,
              make_factory=lambda e, cl: DistillationFactory(
                  e, num_units=1, cycle_ticks=us(11 * RPO * 1.1), decode_service=cl,
                  corr_rounds=2 * D, n_corr=11, return_ticks=us(5.0), initial_store=0),
              title="3) SIX T GATES, undersized factory: large magic-state stall")
 
r = build_and_run(three_cnot_circuit(), num_units=1, d=D, rounds_per_op=RPO,
                  decoder=RelayBPDecoder(),
                  make_metrics=lambda e, cl, ch, f: [DecoderUtilization(cl), ReadyQueueStats(cl)],
                  title="4) Relay-BP decoder + metrics")
print("   metrics:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["metrics"].items()})