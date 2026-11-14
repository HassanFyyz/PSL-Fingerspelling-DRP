"""Serve frozen BEFORE weights or current AFTER weights without swapping files.

Usage: python3 .../serve_comparison.py --profile before --port 8001
       python3 .../serve_comparison.py --profile after --port 8000
Open /web/ on the chosen port. Run in separate terminals for side-by-side tests.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def handler(profile):
    class Handler(SimpleHTTPRequestHandler):
        def translate_path(self, path):
            name = unquote(urlsplit(path).path)
            prefix = "/web/model_geom/"
            if profile == "before" and name.startswith(prefix):
                tail = name[len(prefix):]
                if tail in {p.name for p in (OUT / "model_before").iterdir()}:
                    return str(OUT / "model_before" / tail)
            return super().translate_path(path)

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-PSL-Profile", profile)
            super().end_headers()

        def do_GET(self):
            if urlsplit(self.path).path in ("/web/", "/web/index.html"):
                src = OUT / "index_before.html" if profile == "before" else ROOT / "web/index.html"
                html = src.read_text().replace("<h1>PSL Live Segmentation</h1>",
                    f"<h1>PSL Live Segmentation — {profile.upper()} calibration</h1>")
                data = html.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            super().do_GET()
    return partial(Handler, directory=str(ROOT))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, choices=["before", "after"])
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    print(f"{args.profile.upper()} calibration: http://localhost:{args.port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler(args.profile)).serve_forever()
