#!/usr/bin/env bash
# Restore everything this artifact marks as deleted in .aii/manifest.yaml.
#
# The only removed item is the Python virtual environment, which is fully regenerable
# from pyproject.toml. The archived .npy inputs in data/ are small and are PUBLISHED with
# this folder, so they are not removed; if they are ever absent, `method.py --stage
# checks` regenerates them from pandapower and labels the run REGENERATED.
#
# Run from this artifact's folder (the one holding method.py).
set -euo pipefail

cd "$(dirname "$0")"

echo "== recreating .venv (Python 3.12) =="
uv venv .venv --python=3.12

echo "== installing pinned dependencies from pyproject.toml =="
uv pip install --python=.venv/bin/python \
    numpy==2.4.6 \
    scipy==1.18.1 \
    networkx==3.7 \
    loguru==0.7.3 \
    matplotlib==3.11.2 \
    pandas==2.3.3 \
    pandapower==3.5.6

echo "== verifying the environment =="
.venv/bin/python - <<'PY'
import numpy, scipy, networkx, loguru, matplotlib, pandas, pandapower
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("networkx", networkx.__version__)
print("pandas", pandas.__version__)
print("pandapower", pandapower.__version__)
PY

echo
echo "Environment restored. Next:"
echo "  .venv/bin/python method.py --stage all    # rebuild every result (~5 min, 2 CPUs)"
echo "  .venv/bin/python audit.py                 # re-derive the headline numbers"
