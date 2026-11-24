"""M3: tokens stream as server-sent events with progress, ending in a result frame."""

from research_assistant.pipeline import ProgressLine, analyse
from research_assistant.providers import FakeModel
from research_assistant.sse import format_sse, parse_sse, sse_stream


def test_sse_frames_round_trip(article):
    frames = list(sse_stream(article, FakeModel(first_token_delay=0)))
    assert all(f.startswith("event: ") and f.endswith("\n\n") for f in frames)
    events = parse_sse("".join(frames))
    names = [e for e, _ in events]
    assert names[0] == "start" and names[-1] == "result"
    assert names.count("token") > 3
    chars = [d["chars"] for e, d in events if e == "progress"]
    assert chars == sorted(chars)
    tokens = "".join(d["text"] for e, d in events if e == "token")
    assert len(tokens) == chars[-1]


def test_format_sse_is_one_frame():
    assert format_sse({"event": "progress", "data": {"chars": 3}}) == 'event: progress\ndata: {"chars": 3}\n\n'


def test_progress_line_writes_one_updating_line(article):
    out = []
    analyse(article, FakeModel(first_token_delay=0), on_event=ProgressLine(out.append))
    assert any(s.startswith("\r  attempt 1 |") for s in out)
    assert out[-1].startswith("\r  done | retries 0")
