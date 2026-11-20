"""WSGI entry for the hosted fluency game (PythonAnywhere).

runs fluency_game.route() against a bundle made by
    python fluency_game.py --corpus people --limit 500 --export deploy_bundle
with this file's folder as the bundle root. ratings land in fluency/ here;
pull them back with sync_hosted_game.py.

PythonAnywhere runs one worker process for a free web app, so the in-process
CSV_LOCK in fluency_game is enough to serialise writes.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fluency_game as game  # noqa: E402

game.OUT_DIR.mkdir(parents=True, exist_ok=True)
game.CORPUS = game.FrozenCorpus(ROOT)

STATUS = {200: "200 OK", 400: "400 Bad Request", 404: "404 Not Found",
          500: "500 Internal Server Error"}


def application(environ, start_response):
    method = environ.get("REQUEST_METHOD", "GET")
    path = environ.get("PATH_INFO", "/")
    if environ.get("QUERY_STRING"):
        path += "?" + environ["QUERY_STRING"]
    body = b""
    if method == "POST":
        length = int(environ.get("CONTENT_LENGTH") or 0)
        body = environ["wsgi.input"].read(length) if length else b""
    try:
        status, ctype, data = game.route(method, path, body)
    except Exception as exc:  # keep one bad request from looking like an outage
        print(f"error on {method} {path}: {exc!r}", file=sys.stderr)
        status, ctype, data = 500, "text/plain", b"server error"
    start_response(STATUS.get(status, f"{status} Error"),
                   [("Content-Type", ctype), ("Content-Length", str(len(data))),
                    ("Cache-Control", "no-store" if ctype.startswith(("application/json", "text/html"))
                     else "public, max-age=86400")])
    return [data]
