import importlib


def test_core_modules_import_from_package_context():
    modules = [
        "qecsim.cluster",
        "qecsim.config",
        "qecsim.controllers",
        "qecsim.codes",
        "qecsim.decoders",
        "qecsim.devices",
        "qecsim.engine",
        "qecsim.message",
        "qecsim.metrics",
        "qecsim.orchestrators",
        "qecsim.planner",
        "qecsim.protocols",
        "qecsim.schedulers",
        "qecsim.schemes",
    ]

    for module in modules:
        importlib.import_module(module)
