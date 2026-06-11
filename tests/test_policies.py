#==================================================================
# TESTS FOR POLICY SEAMS (deadlines, routing, switching, round time, idle decode)
#==================================================================
from qecsim.codes import SurfaceCodeModel
from qecsim.config import us
from qecsim.decoders import (CodeRouter, PresetLatencyDecoder, SwitchingDecoder)
from qecsim.frontends.circuit import CircuitFrontend, cnot_plus_two_t_circuit
from qecsim.message import DecodeJob, Operation
from qecsim.schedulers import (EarliestDeadlineScheduler, EnqueueTimeDeadline,
                               ReactionPathDeadline)
from qecsim.wiring import build_and_run


# ---- deadline policies ----------------------------------------------------------------

def test_deadline_policies():
    assert EnqueueTimeDeadline().deadline(None, None, 42, on_reaction_path=True) == 42
    pol = ReactionPathDeadline(slack_ticks=100)
    assert pol.deadline(None, None, 42, on_reaction_path=True) == 42
    assert pol.deadline(None, None, 42, on_reaction_path=False) == 142


def _contended_circuit():
    """Four background CNOTs registered BEFORE a gated T chain, so under FIFO the
    reaction-path windows queue behind the Clifford windows."""
    ops = [Operation(i, f"CNOT(q{2*i+2},q{2*i+3})", (2*i + 2, 2*i + 3), clifford=True)
           for i in range(4)]
    ops.append(Operation(4, "T(q0)", (0,), clifford=False))
    ops.append(Operation(5, "T2(q0)", (0,), clifford=False, gated_by=4))
    return CircuitFrontend(ops).build()


def test_reaction_path_deadline_beats_fifo_under_contention():
    def run(scheduler=None, deadline_policy=None):
        r = build_and_run(_contended_circuit(), num_units=1, d=3, rounds_per_op=11,
                          decoder=PresetLatencyDecoder(5.0), scheduler=scheduler,
                          deadline_policy=deadline_policy, verbose=False)
        return r["chip_done"]                  # ends with the gated T's last round

    fifo = run()
    edf = run(scheduler=EarliestDeadlineScheduler(),
              deadline_policy=ReactionPathDeadline(slack_ticks=us(100.0)))
    assert edf < fifo                          # reaction-path-first shrinks the stall


# ---- decoder routing --------------------------------------------------------------------

def test_code_router_routes_by_code_with_default():
    surface, bb = PresetLatencyDecoder(1.0), PresetLatencyDecoder(2.0)
    router = CodeRouter(default=surface, by_code={"bb": bb})
    assert router.route(DecodeJob(0, 0, 6, code="bb")) is bb
    assert router.route(DecodeJob(0, 0, 6, code="surface")) is surface
    assert router.route(DecodeJob(0, 0, 6)) is surface


def test_custom_router_by_hint():
    class HintRouter:
        def __init__(self, normal, strong):
            self.normal, self.strong = normal, strong
        def route(self, job):
            return self.strong if job.hint == "strong" else self.normal

    weak, strong = PresetLatencyDecoder(1.0), PresetLatencyDecoder(10.0)
    router = HintRouter(weak, strong)
    assert router.route(DecodeJob(0, 0, 6)) is weak
    assert router.route(DecodeJob(0, 0, 6, hint="strong")) is strong
    # the router seam accepts it end to end
    r = build_and_run(cnot_plus_two_t_circuit(), num_units=2, d=3, rounds_per_op=11,
                      router=router, verbose=False)
    assert r["fully_done"] > 0


# ---- timing-level decoder switching (arXiv:2510.25222) ---------------------------------

def test_switching_decoder_latency_mix():
    weak, strong = PresetLatencyDecoder(1.0), PresetLatencyDecoder(10.0)
    job = DecodeJob(0, 0, 6)
    never = SwitchingDecoder(weak, strong, gamma_switch=0.0, handoff_us=0.5)
    assert never.latency(job) == us(1.0) and job.hint is None
    assert never.decode(job).soft_output == 1.0

    job2 = DecodeJob(0, 0, 6)
    always = SwitchingDecoder(weak, strong, gamma_switch=1.0, handoff_us=0.5)
    # weak decode + two handoffs (decoder->decoder messaging) + strong decode
    assert always.latency(job2) == us(1.0) + 2 * us(0.5) + us(10.0)
    assert job2.hint == "strong" and always.switches == 1
    assert always.decode(job2).soft_output == 0.0


def test_switching_decoder_charges_t_comm_weak_on_every_path():
    """The paper's T_comm^weak is paid on EVERY decode (weak path included, its backlog
    recursion has both T_comm terms); default 0 keeps the old latencies exactly."""
    weak, strong = PresetLatencyDecoder(1.0), PresetLatencyDecoder(10.0)
    never = SwitchingDecoder(weak, strong, gamma_switch=0.0, t_comm_weak_us=1.1)
    assert never.latency(DecodeJob(0, 0, 6)) == us(1.1) + us(1.0)
    always = SwitchingDecoder(weak, strong, gamma_switch=1.0, handoff_us=0.5,
                              t_comm_weak_us=1.1)
    assert always.latency(DecodeJob(0, 0, 6)) == us(1.1) + us(1.0) + 2 * us(0.5) + us(10.0)


def test_switching_decoder_end_to_end():
    sw = SwitchingDecoder(PresetLatencyDecoder(1.0), PresetLatencyDecoder(10.0),
                          gamma_switch=0.5, seed=7)
    r = build_and_run(cnot_plus_two_t_circuit(), num_units=2, d=3, rounds_per_op=11,
                      decoder=sw, verbose=False)
    assert r["fully_done"] > 0                 # runs to completion with mixed latencies


# ---- per-code round time (heterogeneous cadence infrastructure) ------------------------

def _memory_op():
    op = Operation(0, "M(q0)", (0,), clifford=True)
    return CircuitFrontend([op]).build()


def test_code_round_time_overrides_global_cadence():
    slow = SurfaceCodeModel(d=3, round_us=2.0)
    r = build_and_run(_memory_op(), num_units=1, code=slow, rounds_per_op=5,
                      round_us=1.1, decoder=PresetLatencyDecoder(0.5), verbose=False)
    assert r["chip_done"] == 5 * us(2.0)       # the CODE's cadence, not the global


def test_global_cadence_is_default():
    r = build_and_run(_memory_op(), num_units=1, d=3, rounds_per_op=5,
                      round_us=1.1, decoder=PresetLatencyDecoder(0.5), verbose=False)
    assert r["chip_done"] == 5 * us(1.1)


# ---- idle-round decoding flag (arXiv:2511.10633: memory rounds need decoding) -----------

def test_idle_rounds_decoded_only_when_enabled():
    def run(flag):
        r = build_and_run(cnot_plus_two_t_circuit(), num_units=2, d=3, rounds_per_op=11,
                          decoder=PresetLatencyDecoder(3.0),
                          decode_idle_rounds=flag, verbose=False)
        return [l for l in r["engine"].log_lines if "mem(" in l]

    assert run(False) == []                    # default: byte-identical, no memory jobs
    assert len(run(True)) > 0                  # flag on: idle rounds load the cluster
