"""Run in this venv:

.venv/bin/python -m pytest tests
"""

import os
import tempfile

import pytest

# bfcl_eval creates result/, score/ and .file_locks/ under its project root at import.
os.environ["BFCL_PROJECT_ROOT"] = tempfile.mkdtemp(prefix="bfcl_project_")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


@pytest.fixture(autouse=True)
def _restore_generation_config(monkeypatch):
    """Tests that install or configure a run's generation config leave the default behind."""
    from domyn_bfcl import config

    monkeypatch.setattr(config, "GENERATION", config.GENERATION)
