#!/usr/bin/env python3
"""Local dev server for the WebGPU demo.

Plain `python3 -m http.server` does NOT set the cross-origin-isolation
headers (COOP/COEP) that modern wasm runtimes need to use
SharedArrayBuffer. Without those, ONNX Runtime Web's threaded wasm
aborts during init with `Aborted()`.

This script serves the current directory with:
  - Cross-Origin-Opener-Policy: same-origin
  - Cross-Origin-Embedder-Policy: credentialless

`credentialless` is the modern variant that lets cross-origin resources
(like the ORT files from jsdelivr) load without forcing the CDN to send
its own CORP header. Chrome 96+, Firefox 110+, Safari 17.4+.

Usage:
    python3 serve.py            # serves on http://localhost:8000
    python3 serve.py 8089       # custom port
"""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import sys


class COIHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        # Allow the demo HTML to fetch the asset files cross-origin too,
        # in case you point ASSET URLs at S3 while serving the HTML locally.
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"serving on http://localhost:{port}  (COOP+COEP=credentialless)")
    ThreadingHTTPServer(("", port), COIHandler).serve_forever()
