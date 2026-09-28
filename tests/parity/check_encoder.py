import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

FIXED_IDS = [0, 1, 50367, 12345, 777, 50283]
LONG_IDS = [(i * 37 + 5) % 50368 for i in range(150)]

TOL = 1e-4
LAYER_TOL = 5e-3


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


def max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def check_embeddings(golden, binary, model_dir, verbose):
    [mine] = run_binary(binary, model_dir, "embeddings", [{"input_ids": FIXED_IDS}])
    if "error" in mine:
        print("ERROR: %s" % mine["error"])
        return 1
    want = golden.embeddings(FIXED_IDS)
    want_flat = want.reshape(-1).tolist()
    want_shape = list(want.shape)
    bad = 0
    if mine["shape"] != want_shape:
        print("SHAPE MISMATCH want=%s got=%s" % (want_shape, mine["shape"]))
        bad += 1
    diff = max_abs_diff(want_flat, mine["values"])
    print("embeddings max abs diff: %.3e (tolerance %.1e)" % (diff, TOL))
    if diff > TOL:
        bad += 1
        if verbose:
            for i, (w, g) in enumerate(zip(want_flat, mine["values"])):
                if abs(w - g) > TOL:
                    print("  first diff at flat index %d: want %r got %r" % (i, w, g))
                    break
    return bad


def check_layer(golden, binary, model_dir, layer_idx, verbose):
    [mine] = run_binary(binary, model_dir, "layer", [{"input_ids": LONG_IDS, "layer": layer_idx}])
    if "error" in mine:
        print("ERROR (layer %d): %s" % (layer_idx, mine["error"]))
        return 1
    want = golden.hidden_after_layer(LONG_IDS, layer_idx)
    want_flat = want.reshape(-1).tolist()
    want_shape = list(want.shape)
    bad = 0
    if mine["shape"] != want_shape:
        print("SHAPE MISMATCH (layer %d) want=%s got=%s" % (layer_idx, want_shape, mine["shape"]))
        bad += 1
    diff = max_abs_diff(want_flat, mine["values"])
    print("layer %d max abs diff: %.3e (tolerance %.1e)" % (layer_idx, diff, LAYER_TOL))
    if diff > LAYER_TOL:
        bad += 1
        if verbose:
            for i, (w, g) in enumerate(zip(want_flat, mine["values"])):
                if abs(w - g) > LAYER_TOL:
                    print("  first diff at flat index %d: want %r got %r" % (i, w, g))
                    break
    return bad


def check_full(golden, binary, model_dir, verbose):
    [mine] = run_binary(binary, model_dir, "full", [{"input_ids": LONG_IDS}])
    if "error" in mine:
        print("ERROR (full encoder): %s" % mine["error"])
        return 1
    want = golden.encoder_output(LONG_IDS)
    want_flat = want.reshape(-1).tolist()
    want_shape = list(want.shape)
    bad = 0
    if mine["shape"] != want_shape:
        print("SHAPE MISMATCH (full) want=%s got=%s" % (want_shape, mine["shape"]))
        bad += 1
    diff = max_abs_diff(want_flat, mine["values"])
    print("full encoder max abs diff: %.3e (tolerance %.1e)" % (diff, LAYER_TOL))
    if diff > LAYER_TOL:
        bad += 1
        if verbose:
            for i, (w, g) in enumerate(zip(want_flat, mine["values"])):
                if abs(w - g) > LAYER_TOL:
                    print("  first diff at flat index %d: want %r got %r" % (i, w, g))
                    break
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(REPO_ROOT, "models", "laya"))
    ap.add_argument("--binary", default=os.path.join(REPO_ROOT, "build", "laya-encoder-probe"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from golden_model import GoldenModel

    golden = GoldenModel(args.model)

    bad = 0
    bad += check_embeddings(golden, args.binary, args.model, args.verbose)
    bad += check_layer(golden, args.binary, args.model, 0, args.verbose)
    bad += check_layer(golden, args.binary, args.model, 1, args.verbose)
    bad += check_full(golden, args.binary, args.model, args.verbose)

    print("model: %s" % args.model)
    print("overall: %s" % ("match" if bad == 0 else "%d mismatches" % bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
