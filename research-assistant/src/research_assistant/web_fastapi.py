"""Optional: the same SSE endpoint on FastAPI. Install with `uv sync --extra web`, run with
`uv run uvicorn research_assistant.web_fastapi:app`. Nothing else imports this module."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .data import article_by_id
from .providers import get_provider
from .sse import sse_stream

app = FastAPI(title="Research assistant")
provider = get_provider()


@app.get("/analyse")
def analyse_get(id: str) -> StreamingResponse:
    article = article_by_id(id)
    if article is None:
        raise HTTPException(404, "unknown article id")
    return StreamingResponse(sse_stream(article, provider), media_type="text/event-stream")
