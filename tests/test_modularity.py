#==================================================================
# MODULARITY CONFORMANCE TESTS
# Two guarantees a researcher extending qecsim relies on:
#   1. every default implementation satisfies its protocol seam, and
#   2. the standard wiring runs with the seams replaced by minimal
#      from-scratch implementations that know nothing about the defaults.
# If a future change breaks an extension point, these tests fail first.
#==================================================================
from qecsim import protocols as P
from qecsim.adapters.pymatching_decoder import PyMatchingDecoder
from qecsim.adapters.stim_device import StimDevice
from qecsim.chip import Chip
from qecsim.cluster import DecoderCluster
from qecsim.codes import (BBCodeModel, ColorCodeModel, SurfaceCodeModel,
                          ToricCodeModel)
from qecsim.config import us
from qecsim.controllers import ModularController
from qecsim.decoders import (CodeRouter, LatencyModelDecoder, ParityDecoder,
                             PresetLatencyDecoder, RelayBPDecoder, SwitchingDecoder)
from qecsim.devices import SyndromeBitDevice, TimingOnlyDevice
from qecsim.engine import Engine
from qecsim.factories import (DistillationFactory, DistillLevel, InfiniteFactory,
                              MultiLevelDistillationFactory)
from qecsim.frontends.circuit import CircuitFrontend, SurgeryIRFrontend
from qecsim.layouts import UniformLayout, ZonedLayout
from qecsim.message import DecodeResult, Decision, Operation, SyndromePayload
from qecsim.metrics import (DecoderUtilization, MagicStateLatency, ReadyQueueStats,
                            WindowLatencyBreakdown)
from qecsim.orchestrators import PauliFrameOrchestrator
from qecsim.planner import CodeRounds, FixedRounds, WindowPlanner
from qecsim.schedulers import (EarliestDeadlineScheduler, EnqueueTimeDeadline,
                               FifoScheduler, ReactionPathDeadline)
from qecsim.schemes import ParallelWindowScheme, SlidingWindowScheme
from qecsim.wiring import build_and_run


def test_default_implementations_satisfy_their_protocols():
    """Every shipped implementation conforms to the seam it plugs into."""
    eng = Engine(verbose=False)
    code = SurfaceCodeModel(d=3)
    ctrl = ModularController(eng)
    orch = PauliFrameOrchestrator(eng)
    cluster = DecoderCluster(eng, PresetLatencyDecoder(1.0), FifoScheduler(),
                             ctrl, orch, num_units=1, code_distance=3)
    chip = Chip(eng, TimingOnlyDevice(), ctrl, cluster, InfiniteFactory(eng),
                round_ticks=us(1.0), code_distance=3)
    pairs = [
        (P.InputFrontend, CircuitFrontend([])),
        (P.InputFrontend, SurgeryIRFrontend("")),
        (P.DeviceModel, TimingOnlyDevice()),
        (P.DeviceModel, SyndromeBitDevice(code)),
        (P.DeviceModel, StimDevice()),
        (P.CodeModel, code),
        (P.CodeModel, BBCodeModel()),
        (P.CodeModel, ColorCodeModel()),
        (P.CodeModel, ToricCodeModel()),
        (P.LayoutModel, UniformLayout(code)),
        (P.LayoutModel, ZonedLayout({}, code)),
        (P.DecodingScheme, SlidingWindowScheme()),
        (P.DecodingScheme, ParallelWindowScheme()),
        (P.ExecutionPlanner, WindowPlanner(SlidingWindowScheme(), UniformLayout(code), 11)),
        (P.RoundsPolicy, FixedRounds(11)),
        (P.RoundsPolicy, CodeRounds()),
        (P.Decoder, LatencyModelDecoder(3)),
        (P.Decoder, PresetLatencyDecoder(1.0)),
        (P.Decoder, ParityDecoder()),
        (P.Decoder, RelayBPDecoder()),
        (P.Decoder, SwitchingDecoder(PresetLatencyDecoder(1.0),
                                     PresetLatencyDecoder(2.0), 0.5)),
        (P.Decoder, PyMatchingDecoder(PresetLatencyDecoder(1.0))),
        (P.Scheduler, FifoScheduler()),
        (P.Scheduler, EarliestDeadlineScheduler()),
        (P.DeadlinePolicy, EnqueueTimeDeadline()),
        (P.DeadlinePolicy, ReactionPathDeadline(0)),
        (P.DecoderRouter, CodeRouter(PresetLatencyDecoder(1.0))),
        (P.Controller, ModularController(eng)),
        (P.Orchestrator, PauliFrameOrchestrator(eng)),
        (P.MagicStateFactory, InfiniteFactory(eng)),
        (P.MagicStateFactory, DistillationFactory(eng, 1, us(1.0), cluster, 1)),
        (P.MagicStateFactory, MultiLevelDistillationFactory(
            eng, [DistillLevel(units=1, d=3)], W_ticks=us(1.0))),
        (P.DecoderService, cluster),
        (P.WorkloadManager, cluster),
        (P.QuantumProcessor, chip),
        (P.Metric, DecoderUtilization(cluster)),
        (P.Metric, ReadyQueueStats(cluster)),
        (P.Metric, WindowLatencyBreakdown(cluster)),
        (P.Metric, MagicStateLatency(InfiniteFactory(eng))),
    ]
    for proto, impl in pairs:
        assert isinstance(impl, proto), f"{type(impl).__name__} fails {proto.__name__}"


# ---- a researcher's from-scratch stack: no defaults, protocol surface only -----------

class MyDevice:
    def begin_operation(self, op): pass
    def round_payload(self, op, r):
        return SyndromePayload(op.id, op.patches[0], r)

class MyCode:
    name = "my-code"
    distance = 3
    def rounds_per_logical_cycle(self): return 3
    def commit_rounds(self): return 3
    def buffer_rounds(self): return 3
    def spatial_nodes(self, n): return 9 * max(1, n)
    def syndrome_bits_per_round(self, n): return 8 * max(1, n)

class MyLayout:
    def __init__(self, code): self.code = code
    name = "my-layout"
    distance = 3
    def code_for_patch(self, p): return self.code
    def code_for_op(self, op): return self.code
    def spatial_nodes_for(self, op): return self.code.spatial_nodes(len(op.qubits))
    def codes(self): return [self.code]

class MyScheme:
    """One window per op committing everything (no chained deps -- wire_deps absent,
    so the planner's chain fallback applies trivially to a single window)."""
    def plan_windows(self, op_id, n_rounds, code):
        return [(1, n_rounds, n_rounds)]
    def data_complete(self, window, rounds_arrived, successor_rounds, memory_rounds,
                      n_rounds, has_successor, op=None, layout=None):
        return rounds_arrived >= window.commit_hi

class MyRounds:
    def rounds_for(self, op, code): return 7

class MyDecoder:
    def __init__(self): self.decodes = 0
    def latency(self, job): return us(0.2)
    def decode(self, job):
        self.decodes += 1
        return DecodeResult(job.op_id, job.window_id, logical_value=1)

class MyScheduler:
    """LIFO -- a genuinely different policy than the default FIFO."""
    def insert(self, queue, job): queue.append(job)
    def pop(self, queue): return queue.pop()

class MyDeadline:
    def deadline(self, op, window, now, on_reaction_path): return now

class MyRouter:
    def __init__(self, decoder): self.decoder = decoder; self.calls = 0
    def route(self, job):
        self.calls += 1
        return self.decoder

class MyController:
    def __init__(self, engine): self.engine = engine
    def relay_syndrome(self, payload, deliver):
        self.engine.schedule(us(0.1), lambda: deliver(payload))
    def relay_instruction(self, decision, deliver):
        self.engine.schedule(us(0.1), lambda: deliver(decision))
    def dec_to_dec_delay(self): return us(0.1)
    def dec_to_orch_delay(self): return us(0.1)

class MyOrchestrator:
    def __init__(self, engine):
        self.engine = engine; self.gates = {}; self.controller = None; self.sink = None
    def connect(self, controller, decision_sink):
        self.controller = controller; self.sink = decision_sink
    def register_gate(self, gated_op_id, gating_op_id):
        self.gates.setdefault(gating_op_id, []).append(gated_op_id)
    def announce_plan(self, plan): pass
    def integrate(self, op, result):
        for g in self.on_result(op, result):
            self.controller.relay_instruction(g, self.sink)
    def on_result(self, op, result):
        return [Decision(g, "Z") for g in self.gates.pop(op.id, [])]

class MyFactory:
    def __init__(self): self.requests = 0
    def request(self, op_id, callback):
        self.requests += 1; callback()
    def shutdown(self): pass

class MyMetric:
    name = "my_events"
    def __init__(self): self.count = 0
    def observe(self, engine): self.count += 1
    def result(self): return self.count


def _gated_ops():
    """CNOT, then a T whose decode gates a second T -- hand-wired, no frontend helpers."""
    a = Operation(0, "CNOT(q0,q1)", (0, 1), clifford=True)
    b = Operation(1, "T(q1)", (1,), clifford=False)
    c = Operation(2, "T2(q1)", (1,), clifford=False, gated_by=1)
    a.patches, b.patches, c.patches = (0, 1), (1,), (1,)
    b.predecessors, c.predecessors = (0,), (1,)
    a.has_successor = b.has_successor = True
    return [a, b, c]


def test_every_seam_accepts_a_from_scratch_implementation():
    """The standard wiring runs end to end with the seams replaced by the from-scratch
    stack above (device, code, layout, scheme, rounds, decoder, router, scheduler,
    deadline policy, controller, orchestrator, factory, metric -- all at once), with
    assertions that each custom piece actually participated."""
    decoder, factory, metric = MyDecoder(), MyFactory(), MyMetric()
    router = MyRouter(decoder)
    r = build_and_run(_gated_ops(), num_units=2,
                      device=MyDevice(), code=MyCode(), layout=MyLayout(MyCode()),
                      scheme=MyScheme(), rounds_policy=MyRounds(),
                      decoder=decoder, router=router, scheduler=MyScheduler(),
                      deadline_policy=MyDeadline(),
                      make_controller=MyController,
                      make_orchestrator=MyOrchestrator,
                      factory=factory,
                      make_metrics=lambda e, cl, ch, f: [metric],
                      verbose=False)
    chip = r["chip"]
    assert len(chip.done_bodies) == 3          # all ops ran, INCLUDING the gated T
    assert decoder.decodes >= 3                # the custom decoder decoded every window
    assert router.calls >= 3                   # routed per job
    assert factory.requests == 2               # both T gates drew a state
    assert metric.count > 0                    # the custom metric observed events
    assert r["fully_done"] > r["chip_done"] >= 0


def test_make_cluster_and_workload_manager_protocol():
    """make_cluster swaps the workload manager; the default satisfies the protocol."""
    built = {}
    def make_cluster(engine, decoder, scheduler, controller, orchestrator):
        c = DecoderCluster(engine, decoder, scheduler, controller, orchestrator,
                           num_units=4, code_distance=3)
        built["cluster"] = c
        return c
    r = build_and_run(_gated_ops(), make_cluster=make_cluster, verbose=False)
    assert r["cluster"] is built["cluster"]
    assert isinstance(built["cluster"], P.WorkloadManager)
    assert built["cluster"].num_units == 4     # OUR cluster ran, not a default one


def test_planner_parameter_swaps_the_planning_algorithm():
    """planner= replaces the ExecutionPlanner in the standard wiring."""
    class CountingPlanner:
        def __init__(self, inner): self.inner = inner; self.calls = 0
        def plan(self, ops):
            self.calls += 1
            return self.inner.plan(ops)
    code = SurfaceCodeModel(d=3)
    planner = CountingPlanner(WindowPlanner(SlidingWindowScheme(), UniformLayout(code), 11))
    r = build_and_run(_gated_ops(), d=3, rounds_per_op=11, planner=planner, verbose=False)
    assert planner.calls == 1                  # OUR planner produced the plan
    assert r["fully_done"] > 0
