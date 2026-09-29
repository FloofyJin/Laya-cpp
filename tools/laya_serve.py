import argparse
import ctypes
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

lib = None
agent = None


def load_lib(lib_path):
    global lib
    lib = ctypes.CDLL(lib_path)
    lib.laya_agent_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.laya_agent_load.restype = ctypes.c_void_p
    lib.laya_agent_free.argtypes = [ctypes.c_void_p]
    lib.laya_agent_free.restype = None
    lib.laya_predict.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.laya_predict.restype = ctypes.c_void_p
    lib.laya_free_string.argtypes = [ctypes.c_void_p]
    lib.laya_free_string.restype = None


def load_agent(model_dir, chip):
    global agent
    err = ctypes.create_string_buffer(512)
    agent = lib.laya_agent_load(model_dir.encode("utf-8"), chip.encode("utf-8"), err, len(err))
    if not agent:
        raise RuntimeError("laya_agent_load failed: %s" % err.value.decode("utf-8", "replace"))


def predict(request_json_text):
    err = ctypes.create_string_buffer(512)
    ptr = lib.laya_predict(agent, request_json_text.encode("utf-8"), err, len(err))
    if not ptr:
        raise RuntimeError(err.value.decode("utf-8", "replace"))
    try:
        text = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
    finally:
        lib.laya_free_string(ptr)
    return text


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self._cors()
            self.end_headers()

    def do_POST(self):
        if self.path != "/predict":
            self.send_response(404)
            self._cors()
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        try:
            result = predict(body)
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(result.encode("utf-8"))
        except Exception as e:
            self.send_response(500)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def log_message(self, fmt, *args):
        sys.stderr.write("laya-serve: %s\n" % (fmt % args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(REPO_ROOT, "models", "laya"))
    ap.add_argument("--lib", default=os.path.join(REPO_ROOT, "build", "liblaya_core.so"))
    ap.add_argument("--chip", default="hip")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()

    load_lib(args.lib)
    print("laya-serve: loading model (chip=%s)..." % args.chip, flush=True)
    load_agent(args.model, args.chip)
    print("laya-serve: ready on http://localhost:%d (POST /predict, GET /health)" % args.port, flush=True)

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        lib.laya_agent_free(agent)


if __name__ == "__main__":
    main()
