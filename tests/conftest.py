from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep settings tests deterministic and independent of any local config.yaml/.env
# that a developer may have created. Tests drive configuration purely via env vars.
os.environ.setdefault("CONFIG_PATH", str(ROOT / "tests" / "_no_such_config.yaml"))
os.environ.setdefault("ENV_FILE", str(ROOT / "tests" / "_no_such.env"))
