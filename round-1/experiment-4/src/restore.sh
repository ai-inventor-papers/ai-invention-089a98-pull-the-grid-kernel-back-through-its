#!/usr/bin/env bash
# Restore everything this artifact marks as removed, then verify the results.
#
# The only heavy path this workspace removes is `.venv/` (1.3 GB). Everything
# else — data/, results/, figures/, figure_specs/ — is kept on the run's volume
# and, except for .venv, is published in the repository.
set -u

cd "$(dirname "$0")"

echo "=== rebuilding the Python environment (.venv/) ==="
# .venv/ was removed as regenerable; this is the command that rebuilds it
uv venv .venv --python=3.12
uv pip install --python .venv/bin/python -r requirements-lock.txt

echo "=== clearing the regenerable bytecode cache (__pycache__/) ==="
rm -rf __pycache__

echo "=== verifying the environment ==="
.venv/bin/python -c "import numpy, scipy, pandapower, networkx, loguru; \
print('numpy', numpy.__version__, 'scipy', scipy.__version__, \
      'pandapower', pandapower.__version__)"

echo "=== staged tests T1-T6 (~4 min) ==="
.venv/bin/python test_stages.py

echo "=== the full sweep, or a reuse of the kept panels (~1 or 9 min) ==="
# results/screen_D.json is KEPT, so method.py reuses the completed panels by
# default.  Set RESUME_PANELS=0 to recompute the whole sweep from scratch.
.venv/bin/python method.py

echo "=== independent re-derivation and placebo tests (~5 min) ==="
.venv/bin/python audit.py

echo "=== figures ==="
.venv/bin/python make_figures.py

echo
echo "done. Check results/screen_D.json, results/protocol_checks.json and"
echo "figures/ — the expected numbers are listed in reproducibility.md section 6."
