import json

import pytest

from research_assistant.data import article_by_id
from research_assistant.providers import extract


@pytest.fixture
def article():
    return article_by_id("a01")


@pytest.fixture
def good_reply(article):
    return json.dumps(extract(article["id"], article["text"]))
