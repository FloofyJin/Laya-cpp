import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SAMPLE_LIMIT = 4096


class Golden:
    def __init__(self, model_dir):
        from safetensors import safe_open

        self.f = safe_open(os.path.join(model_dir, "model.safetensors"), framework="numpy")
        self.names = sorted(self.f.keys())

    def dtype_shape(self, name):
        s = self.f.get_slice(name)
        return s.get_dtype(), list(s.get_shape())

    def values(self, name, start, count):
        import numpy as np

        arr = self.f.get_tensor(name).astype(np.float32).reshape(-1)
        return arr[start:start + count].tolist()


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def run_binary(binary, model_dir, mode, rows):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        path = f.name
    try:
        out = subprocess.run([binary, "--model", model_dir, mode, path], check=True, capture_output=True, text=True)
    finally:
        os.unlink(path)
    lines = [json.loads(line) for line in out.stdout.splitlines() if line.strip()]
    if len(lines) != len(rows):
        raise RuntimeError("binary returned %d results for %d inputs (stderr: %s)" %
                           (len(lines), len(rows), out.stderr))
    return lines


def check_names(golden, names_out, verbose):
    mine = {n["name"]: (n["dtype"], n["shape"]) for n in names_out["names"]}
    missing = sorted(set(golden.names) - set(mine))
    extra = sorted(set(mine) - set(golden.names))
    bad = 0
    if missing or extra:
        bad += 1
        print("NAME SET MISMATCH missing=%s extra=%s" % (missing[:10], extra[:10]))
    for name in golden.names:
        if name not in mine:
            continue
        want = golden.dtype_shape(name)
        if mine[name] != want:
            bad += 1
            if verbose or bad <= 10:
                print("METADATA MISMATCH %s want=%s got=%s" % (name, want, mine[name]))
    return bad


def check_values(golden, binary, model_dir, verbose):
    requests = []
    for name in golden.names:
        _, shape = golden.dtype_shape(name)
        n = numel(shape)
        if n <= SAMPLE_LIMIT:
            requests.append({"name": name, "start": 0, "count": n})
        else:
            requests.append({"name": name, "start": 0, "count": SAMPLE_LIMIT})
            requests.append({"name": name, "start": n - SAMPLE_LIMIT, "count": SAMPLE_LIMIT})

    results = run_binary(binary, model_dir, "values", requests)
    bad = 0
    for req, mine in zip(requests, results):
        name = req["name"]
        if "error" in mine:
            bad += 1
            print("ERROR on %s start=%d count=%d: %s" % (name, req["start"], req["count"], mine["error"]))
            continue
        want_values = golden.values(name, req["start"], req["count"])
        want_dtype, want_shape = golden.dtype_shape(name)
        if mine["dtype"] != want_dtype or mine["shape"] != want_shape or mine["values"] != want_values:
            bad += 1
            if verbose or bad <= 10:
                print("VALUE MISMATCH %s start=%d count=%d" % (name, req["start"], req["count"]))
                if mine["dtype"] != want_dtype:
                    print("  dtype want=%s got=%s" % (want_dtype, mine["dtype"]))
                if mine["shape"] != want_shape:
                    print("  shape want=%s got=%s" % (want_shape, mine["shape"]))
                if mine["values"] != want_values:
                    for i, (w, g) in enumerate(zip(want_values, mine["values"])):
                        if w != g:
                            print("  first diff at local index %d: want %r got %r" % (i, w, g))
                            break
    return bad, len(requests)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(REPO_ROOT, "models", "laya"))
    ap.add_argument("--binary", default=os.path.join(REPO_ROOT, "build", "laya-dump-tensor"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    golden = Golden(args.model)

    [names_out] = run_binary(args.binary, args.model, "names", [{}])
    name_bad = check_names(golden, names_out, args.verbose)
    value_bad, value_total = check_values(golden, args.binary, args.model, args.verbose)

    print("model: %s" % args.model)
    print("tensor count: %d" % len(golden.names))
    print("names+metadata: %s" % ("match" if name_bad == 0 else "%d mismatches" % name_bad))
    print("value samples: %d/%d match" % (value_total - value_bad, value_total))
    sys.exit(1 if name_bad or value_bad else 0)


if __name__ == "__main__":
    main()
