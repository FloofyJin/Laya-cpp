import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TOL = 1e-12

CASES = [
    ("shipped laya config", None),
    ("NaN in temperature", '{"temperature": [NaN, 1.25, 1.98]}'),
    ("Infinity and -Infinity in temperature", '{"temperature": [Infinity, -Infinity, 1.98]}'),
    ("null in temperature", '{"temperature": [null, 1.25, 1.98]}'),
    ("spacing around NaN", '{"temperature":[ NaN ,-Infinity,2.0 ]}'),
    ("NaN in temperature_by_options", '{"temperature": [1.6, 1.25, 1.98], '
                                      '"temperature_by_options": {"choice:2": NaN, "noul:2": 1.98}}'),
    ("clamping still applies", '{"temperature": [0.1, 9.0, 0.5], '
                               '"temperature_by_options": {"choice:11+": 0.1006, "choice:2": 5.0}}'),
    ("NaN inside a key is left alone", '{"temperature": [1.6, 1.25, 1.98], '
                                       '"temperature_by_options": {"NaN:2": 1.7, "Infinity": NaN}}'),
    ("escaped quote inside a key", '{"temperature": [1.6, 1.25, 1.98], '
                                   '"temperature_by_options": {"a\\"NaN": Infinity}}'),
    ("NaN inside a string value", '{"note": "NaN and -Infinity \\\\", "temperature": [NaN, 1.25, 1.98]}'),
    ("missing temperature keys", '{"max_len": 512}'),
    ("lowercase nan is still invalid", '{"temperature": [nan, 1.25, 1.98]}'),
]


def expected(text):
    from laya.common import clamp_temperature

    try:
        cfg = json.loads(text)
    except ValueError as e:
        return {"error": str(e)}
    temps = [clamp_temperature(t) for t in cfg.get("temperature", [1.0, 1.0, 1.0])]
    by_options = {k: clamp_temperature(v) for k, v in cfg.get("temperature_by_options", {}).items()}
    return {"temperature": temps, "temperature_by_options": dict(sorted(by_options.items()))}


def run_probe(binary, text):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write(text)
        path = f.name
    try:
        out = subprocess.run([binary, path], check=True, capture_output=True, text=True)
    finally:
        os.unlink(path)
    return json.loads(out.stdout)


def compare(want, got):
    if "error" in want or "error" in got:
        if "error" in want and "error" in got:
            return []
        return ["want %r got %r" % (want, got)]
    diffs = []
    for i, (w, g) in enumerate(zip(want["temperature"], got["temperature"])):
        if abs(w - g) > TOL:
            diffs.append("temperature[%d]: want %r got %r" % (i, w, g))
    if list(want["temperature_by_options"]) != list(got["temperature_by_options"]):
        diffs.append("temperature_by_options keys: want %r got %r" %
                     (list(want["temperature_by_options"]), list(got["temperature_by_options"])))
    else:
        for k, w in want["temperature_by_options"].items():
            g = got["temperature_by_options"][k]
            if abs(w - g) > TOL:
                diffs.append("temperature_by_options[%r]: want %r got %r" % (k, w, g))
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(REPO_ROOT, "models", "laya"))
    ap.add_argument("--binary", default=os.path.join(REPO_ROOT, "build", "laya-temperature-probe"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    passed = 0
    for name, text in CASES:
        if text is None:
            with open(os.path.join(args.model, "rl_agent_config.json"), encoding="utf-8") as f:
                text = f.read()
        want = expected(text)
        got = run_probe(args.binary, text)
        diffs = compare(want, got)
        if diffs:
            print("FAIL %s" % name)
            for d in diffs:
                print("  " + d)
        else:
            passed += 1
            print("ok   %s" % name)
            if args.verbose:
                print("     %s" % json.dumps(got))

    print("cases: %d/%d match" % (passed, len(CASES)))
    sys.exit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
