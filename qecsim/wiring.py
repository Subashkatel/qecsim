
from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

from .config import SimConfig, us, fmt
from .engine import Engine
from .codes import SurfaceCodeModel
from .devices import TimingOnlyDevice
from .orchestrators import PauliFrameOrchestrator
from .schedulers import FifoScheduler
from .cluster import DecoderCluster
from .factories import InfiniteFactory
from .chip import Chip
from .planner import WindowPlanner

if TYPE_CHECKING:                      # type-only; defaults are built from the concrete imports above
    from .message import Operation
    from .protocols import (MagicStateFactory, Decoder, Controller, Orchestrator,
                           Scheduler, DeviceModel, CodeModel, DecodingScheme,
                           LayoutModel, InputFrontend, RoundsPolicy,
                           DecoderRouter, DeadlinePolicy, ExecutionPlanner)

# ============================================================================================
# WIRING
# This module defines the default wiring of the qecsim package: how the components are connected
# together, and the default policies for the planner and the decoder. The build_and_run()
# function is the main entry point: it assembles the standard pipeline (engine, components, chip),
# runs it, and returns results 
# ============================================================================================

def build_and_run(ops: Optional[list[Operation]] = None, num_units: Optional[int] = None,
                  d: int = 3, rounds_per_op: Optional[int] = None,
                  rounds_policy: Optional["RoundsPolicy"] = None,
                  round_us: Optional[float] = None,
                  factory: Optional[MagicStateFactory] = None,
                  make_factory: Optional[Callable[[Engine, "DecoderCluster"],
                                                   MagicStateFactory]] = None,
                  decoder: Optional[Decoder] = None,
                  decoders: Optional[dict] = None,
                  controller: Optional[Controller] = None,
                  make_controller: Optional[Callable[[Engine], "Controller"]] = None,
                  orchestrator: Optional[Orchestrator] = None,
                  make_orchestrator: Optional[Callable[[Engine], "Orchestrator"]] = None,
                  scheduler: Optional[Scheduler] = None,
                  router: Optional["DecoderRouter"] = None,
                  deadline_policy: Optional["DeadlinePolicy"] = None,
                  decode_idle_rounds: bool = False,
                  max_idle_rounds: Optional[int] = None,
                  gates_start_on_round_boundaries: bool = False,
                  unit_pools: Optional[dict] = None,
                  device: Optional[DeviceModel] = None,
                  make_cluster: Optional[Callable] = None,
                  planner: Optional["ExecutionPlanner"] = None,
                  make_chip: Optional[Callable] = None,
                  make_metrics: Optional[Callable] = None,
                  code: Optional[CodeModel] = None,
                  scheme: Optional["DecodingScheme"] = None,
                  layout: Optional["LayoutModel"] = None,
                  frontend: Optional["InputFrontend"] = None,
                  config: Optional[SimConfig] = None,
                  verbose: bool = True, title: str = "") -> dict:
    """Assemble the standard pipeline (engine, components, chip), run it, and return results plus any metrics. Every collaborator is optional and swappable."""
    engine = Engine(verbose=verbose)
    if title:
        print("=" * 78)
        print(title)
        print("=" * 78)

    # CONFIG seam: scalar knobs come from SimConfig; an explicit argument still wins.
    cfg = config or SimConfig()
    if num_units is None:     num_units = cfg.num_units
    if rounds_per_op is None: rounds_per_op = cfg.rounds_per_op
    if round_us is None:      round_us = cfg.round_us

    # INPUT seam: take a frontend (Circuit today; OpenQASM / Surgery IR later) or a
    # ready-made operation list. The frontend is the place new input formats plug in.
    if frontend is not None:
        ops = frontend.build()
    if ops is None:
        raise ValueError("provide either ops=<list[Operation]> or frontend=<InputFrontend>")

    # CODE seam: a CodeModel, or a surface code of distance d (back-compatible default).
    # If a heterogeneous LAYOUT is given without an explicit code, the layout's first code is
    # the representative used to size the default device/decoder.
    if code is None:
        code = layout.codes()[0] if layout is not None else SurfaceCodeModel(d=d)

    # Every part is a swap point: pass your own, or get the config-built default.
    if device is None:        device = TimingOnlyDevice()
    if decoder is None:       decoder = cfg.make_decoder(code)
    if make_controller is not None:    controller = make_controller(engine)   # engine-dependent swap
    elif controller is None:           controller = cfg.make_controller(engine)
    if make_orchestrator is not None:  orchestrator = make_orchestrator(engine)   # engine-dependent swap
    elif orchestrator is None:         orchestrator = PauliFrameOrchestrator(engine)
    if scheduler is None:     scheduler = FifoScheduler()

    # LINKS: one LinkModel (the fabric price list) is shared by the controller and the
    # cluster, so the fabric cannot disagree with itself. A controller built above
    # carries one; a custom controller without a .links attribute gets the config-built
    # fabric for the cluster's two hops (dd, do).
    links = getattr(controller, "links", None)
    if links is None:
        links = cfg.make_links()

    # WORKLOAD-MANAGER seam: the default DecoderCluster, or a custom manager via
    # make_cluster(engine, decoder, scheduler, controller, orchestrator). A custom manager
    # must satisfy the WorkloadManager protocol (protocols.py); if it does not expose the
    # scheme/layout/rounds_policy attributes the default planner reads, pass planner= too.
    if make_cluster is not None:
        cluster = make_cluster(engine, decoder, scheduler, controller, orchestrator)
    else:
        cluster = DecoderCluster(engine, decoder, scheduler, controller, orchestrator,
                                 num_units=num_units, rounds_per_op=rounds_per_op, code=code,
                                 scheme=scheme, layout=layout, decoders=decoders,
                                 rounds_policy=rounds_policy, router=router,
                                 deadline_policy=deadline_policy, links=links,
                                 unit_pools=unit_pools)

    # FACTORY seam: an explicit factory, or a make_factory(engine, cluster) hook for
    # factories that must route their correction decodes through THIS cluster (the
    # DistillationFactory / MultiLevelDistillationFactory need that reference), or the
    # cluster-free InfiniteFactory default. The cluster is built first so the hook can
    # see it -- that is the required information flow, not a hidden coupling.
    if factory is None:
        factory = make_factory(engine, cluster) if make_factory is not None \
                  else InfiniteFactory(engine)

    # QPU seam: a custom processor via make_chip(engine, device, controller, cluster, factory,
    # round_ticks, code), or the default Chip. It must satisfy QuantumProcessor. Note the
    # per-op round count is not passed: it belongs to the cluster's ROUNDS policy
    # (cluster.rounds_for(op)), the single source of truth a chip should read.
    if make_chip is not None:
        chip = make_chip(engine, device, controller, cluster, factory,
                         us(round_us), code)
    else:
        chip = Chip(engine, device, controller, cluster, factory,
                    round_ticks=us(round_us),
                    code_distance=code.distance, decode_idle_rounds=decode_idle_rounds,
                    max_idle_rounds=max_idle_rounds,
                    gates_start_on_round_boundaries=gates_start_on_round_boundaries)
    # Dependency inversion: the cluster gets only the callbacks it needs, not the chip object.
    orchestrator.connect(controller, chip.on_decision)  # orchestrator owns the conditional return path
    cluster.on_workload_complete = factory.shutdown    # lifecycle: stop the factory when done

    # optional metrics: make_metrics(engine, cluster, chip, factory) -> list[Metric]
    if make_metrics is not None:
        for m in make_metrics(engine, cluster, chip, factory):
            engine.add_metric(m)

    # register T-gate gating relationships with the orchestrator
    for op in ops:
        if op.gated_by is not None:
            orchestrator.register_gate(gated_op_id=op.id, gating_op_id=op.gated_by)

    # ORCHESTRATOR, offline phase (arXiv:2511.10633 Sec III, job 2): compile the operation DAG
    # into an execution plan -- the decoding windows, job sequence, and dependency graph -- and
    # hand it to the decoder cluster's workload manager AHEAD OF TIME (0 simulated ticks, off the
    # reaction path). The cluster then only runs the queue against this plan at runtime.
    # PLANNER seam: pass planner=<ExecutionPlanner> to swap the planning algorithm; the
    # default WindowPlanner is built from the cluster's own scheme/layout/rounds policy.
    if planner is None:
        planner = WindowPlanner(cluster.scheme, cluster.layout, cluster.rounds_policy)
    for op in ops:
        cluster.register_op(op)
    plan = planner.plan(ops)
    orchestrator.announce_plan(plan)            # Orchestrator: "sending ... to the cluster"
    cluster.load_execution_plan(plan)           # DecoderClstr: "received execution plan ..."

    chip.load(ops)
    engine.run()

    # ---- summary (printed only when verbose, like the trace itself) ----
    chip_done = chip.last_finish_time
    last_event = engine.now
    if verbose:
        print("-" * 78)
        print(f"SUMMARY ({num_units} decoder unit(s)):")
        print(f"  chip finished all physical work : {fmt(chip_done)}")
        print(f"  decoder fully finished          : {fmt(last_event)}")
        print(f"  reaction tail (chip->fully done): {fmt(last_event - chip_done)}")
        peak_q = max((q for _, q in getattr(cluster, "queue_log", [])), default=0)
        print(f"  peak ready-queue length         : {peak_q}")
        # TODO: verify the the cluster peak payload is calculated correctly ---
        # NOTE: cluster.peak_payloads (syndrome RAM high-water, arXiv:2511.10633 Sec III)
        # is still measured but deliberately NOT printed: it counts payloads RESIDENT under
        # the per-op release rule, not the minimal live set a cluster must provision, and
        # we don't want to show a storage number until that accounting is verified.
        # factory stats, duck-typed: any factory exposing the scalar counters gets the
        # lines (a custom factory without them simply prints nothing extra).
        if isinstance(getattr(factory, "produced", None), int):
            print(f"  magic states produced           : {factory.produced}")
            print(f"  peak magic states in storage    : {factory.peak_in_flight}")
            print(f"  total magic-state supply stall  : {fmt(factory.total_stall)}")
        print()
    return {"engine": engine, "cluster": cluster, "factory": factory, "chip": chip,
            "orchestrator": orchestrator, "controller": controller,
            "chip_done": chip_done, "fully_done": last_event,
            "metrics": engine.metric_results()}