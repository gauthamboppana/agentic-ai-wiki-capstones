import pytest

from reviewer.app import Reviewer
from reviewer.model import ScriptedModel


@pytest.fixture
def make_reviewer(tmp_path):
    """Build a Reviewer on a fresh copy of the seeded repo. Same name, same work directory."""
    def make(name="main", **kw):
        return Reviewer(tmp_path / name, kw.pop("model", None) or ScriptedModel(), **kw)
    return make
