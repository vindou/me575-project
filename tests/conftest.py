import pytest
from hypothesis import settings

import cubesat_sim  # noqa: F401  (turns on float64 before any test builds arrays)
from cubesat_sim import load_config
from helpers import CFG_PATH

# JAX compiles on the first call, so per-example deadlines are meaningless.
settings.register_profile("default", deadline=None, max_examples=40)
settings.load_profile("default")


@pytest.fixture(scope="session")
def cfg():
    return load_config(CFG_PATH)
