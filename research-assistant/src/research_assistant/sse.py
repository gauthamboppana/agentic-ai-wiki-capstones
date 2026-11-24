"""M3: server-sent events. A stdlib HTTP endpoint by default; FastAPI is an optional extra (web_fastapi.py)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .data import article_by_id
from .pipeline import run
from .providers import Provider, get_provider


def format_sse(event: dict) -> str:
    """One SSE frame: an `event:` line, a `data:` line of JSON, and a blank line to end it."""
    return f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"


def sse_stream(article: dict, provider: Provider, **kwargs) -> Iterator[str]:
    """Every pipeline event as an SSE frame. Ends with a `result` or an `error` frame, never an exception."""
    for event in run(article, provider, **kwargs):
        yield format_sse(event)


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """The client side, for tests and scripts: split a stream back into (event, data) pairs."""
    out = []
    for frame in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in frame.splitlines() if ": " in line)
        out.append((fields.get("event", "message"), json.loads(fields.get("data", "null"))))
    return out


PAGE = """<!doctype html><meta charset="utf-8"><title>Research assistant</title>
<p>Article id: <input id="a" value="a04"> <button onclick="go()">Analyse</button></p>
<p id="p">idle</p><pre id="o"></pre>
<script>
function go(){const o=document.getElementById('o'),p=document.getElementById('p');o.textContent='';
const es=new EventSource('/analyse?id='+document.getElementById('a').value);
es.addEventListener('progress',e=>{const d=JSON.parse(e.data);p.textContent='attempt '+d.attempt+' | '+d.chars+' chars'});
es.addEventListener('invalid',e=>{o.textContent+='repair: '+JSON.parse(e.data).error+'\\n'});
es.addEventListener('result',e=>{o.textContent+=JSON.stringify(JSON.parse(e.data),null,2);es.close()});
es.addEventListener('error',e=>{if(e.data)o.textContent+=JSON.parse(e.data).message;es.close()});}
</script>"""


def make_handler(provider: Provider):
    class Handler(BaseHTTPRequestHandler):
        def _stream(self, article: dict) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for frame in sse_stream(article, provider):
                self.wfile.write(frame.encode())
                self.wfile.flush()

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if url.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/analyse":
                article = article_by_id(parse_qs(url.query).get("id", [""])[0])
                if article is None:
                    self.send_error(404, "unknown article id")
                else:
                    self._stream(article)
            else:
                self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/analyse":
                return self.send_error(404)
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                article = {"id": str(body.get("id", "custom")), "title": str(body.get("title", "")), "text": str(body["text"])}
            except (ValueError, KeyError):
                return self.send_error(400, 'send JSON like {"id": "x", "title": "...", "text": "..."}')
            self._stream(article)

        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8000, provider: Provider | None = None) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(provider or get_provider()))
    print(f"Serving on http://{host}:{port}/  (GET /analyse?id=a04 streams events)")
    server.serve_forever()
