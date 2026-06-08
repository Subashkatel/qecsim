# qecsim

A discrete-event simulator for quantum error correction.

## Requirements

- Python 3.9 or newer. The core has **no required dependencies**.
- Optional, only for the real-decoder adapters in `qecsim/adapters/`:
  `stim` and `pymatching` (`pip install stim pymatching`).

## Running the examples

The examples live in [`examples/run_examples.py`](examples/run_examples.py) and
run straight from the source tree (no install needed).

Run **all** of them:

```bash
python examples/run_examples.py
```

To build your own circuit instead of using a preset, create `Operation`s (or
use a frontend in [`qecsim/frontends/`](qecsim/frontends/)) and pass the list as
the first argument to `build_and_run`.

## Running the tests

```bash
python -m pytest tests/
```
