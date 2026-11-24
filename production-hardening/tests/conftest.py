import pytest

from hardening.agent import Config
from hardening.evals import run_suite


@pytest.fixture(scope="session")
def hardened():
    return run_suite(Config())


@pytest.fixture(scope="session")
def unhardened():
    return run_suite(Config.off())
