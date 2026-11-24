import pytest

from pdfrag.corpus import load_chunks, load_documents
from pdfrag.evaluate import evaluate
from pdfrag.golden import load_golden
from pdfrag.pipeline import Pipeline


@pytest.fixture(scope="session")
def docs():
    return load_documents()


@pytest.fixture(scope="session")
def chunks():
    return load_chunks()


@pytest.fixture(scope="session")
def golden():
    return load_golden()


@pytest.fixture(scope="session")
def pipeline(chunks):
    return Pipeline(chunks)


@pytest.fixture(scope="session")
def report(pipeline, golden):
    return evaluate(pipeline, golden)
