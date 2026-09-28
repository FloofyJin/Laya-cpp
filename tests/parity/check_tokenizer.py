import argparse
import json
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cases

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_model(model):
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model, allow_patterns=["rl_agent_config.json", "tokenizer/*"])


class Golden:
    def __init__(self, model_dir):
        from laya.agent import Agent, _fix_tokenizer_config, _load_tokenizer
        from laya.common import encode_text

        _fix_tokenizer_config(model_dir)
        with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
            self.cfg = json.load(f)
        self.tok = _load_tokenizer(os.path.join(model_dir, "tokenizer"), self.cfg)
        self.agent_cls = Agent
        self.encode_text = encode_text

    def ids(self, text):
        return list(self.encode_text(self.tok, text, add_special_tokens=False)["input_ids"])

    def request(self, req):
        questions = req.get("questions")
        state = req.get("state")
        try:
            if not isinstance(questions, dict):
                raise TypeError("questions must be a dict")
            if state is None:
                raise TypeError("state must not be None")
            ids = list(questions.keys())
            for qid in ids:
                self.agent_cls._check_question(qid, questions[qid])
            internal = {qid: self.agent_cls._to_internal(questions[qid]) for qid in ids}
            if not ids:
                return {"items": []}
            fake = SimpleNamespace(tok=self.tok, cfg=self.cfg)
            items = self.agent_cls._encode_state(
                fake, state, ids, internal, max_len=req.get("max_len"), head_max_len=req.get("head_max_len")
            )
        except Exception as e:
            return {"error": "%s: %s" % (type(e).__name__, e)}
        return {"items": [
            {"qid": qid, "ids": list(it["ids"]), "markers": list(it["markers"]), "qtype": it["qtype"],
             "options": it["options"]}
            for qid, it in zip(ids, items)
        ]}


def run_binary(binary, model_dir, mode, rows):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        path = f.name
    try:
        out = subprocess.run([binary, "--model", model_dir, mode, path], check=True, capture_output=True, text=True)
    finally:
        os.unlink(path)
    lines = [json.loads(line) for line in out.stdout.splitlines() if line.strip()]
    if len(lines) != len(rows):
        raise RuntimeError("binary returned %d results for %d inputs" % (len(lines), len(rows)))
    return lines


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def show_tokens(golden, ids, at):
    lo = max(0, at - 3)
    return [golden.tok.convert_ids_to_tokens(i) for i in ids[lo:at + 4]]


def compare_texts(golden, texts, got, verbose):
    bad = 0
    for text, mine in zip(texts, got):
        want = golden.ids(text)
        if mine != want:
            bad += 1
            if verbose or bad <= 10:
                at = first_diff(want, mine if isinstance(mine, list) else [])
                print("TEXT MISMATCH %r" % text[:120])
                if isinstance(mine, list):
                    print("  first diff at %d: want %s got %s" % (at, show_tokens(golden, want, at), show_tokens(golden, mine, at)))
                else:
                    print("  got %s" % mine)
    return bad


def compare_requests(golden, reqs, got, verbose):
    bad = 0
    for req, mine in zip(reqs, got):
        want = golden.request(req)
        if "error" in want or "error" in mine:
            if ("error" in want) != ("error" in mine):
                bad += 1
                print("ERROR MISMATCH want=%s got=%s" % (want.get("error"), mine.get("error")))
            continue
        if want != mine:
            bad += 1
            if verbose or bad <= 10:
                print("REQUEST MISMATCH state=%r" % (str(req.get("state"))[:80],))
                for w, m in zip(want["items"], mine["items"]):
                    if w != m:
                        at = first_diff(w["ids"], m["ids"])
                        print("  qid=%s len want=%d got=%d markers want=%s got=%s options want=%s got=%s" % (
                            w["qid"], len(w["ids"]), len(m["ids"]), w["markers"], m["markers"], w["options"], m["options"]))
                        if w["ids"] != m["ids"]:
                            print("  first id diff at %d: want %s got %s" % (
                                at, show_tokens(golden, w["ids"], at), show_tokens(golden, m["ids"], at)))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--binary", default=os.path.join(REPO_ROOT, "build", "laya-tokenize"))
    ap.add_argument("--fuzz", type=int, default=2000)
    ap.add_argument("--fuzz-requests", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    model_dir = resolve_model(args.model)
    golden = Golden(model_dir)

    texts = cases.TEXTS + cases.fuzz_texts(args.seed, args.fuzz)
    reqs = cases.request_cases() + cases.fuzz_requests(args.seed, args.fuzz_requests)

    text_bad = compare_texts(golden, texts, run_binary(args.binary, model_dir, "texts", texts), args.verbose)
    req_bad = compare_requests(golden, reqs, run_binary(args.binary, model_dir, "requests", reqs), args.verbose)
    errors = sum(1 for r in reqs if "error" in golden.request(r))

    print("model: %s" % model_dir)
    print("raw tokenizer: %d/%d match" % (len(texts) - text_bad, len(texts)))
    print("laya sequences: %d/%d match (%d expected-error cases)" % (len(reqs) - req_bad, len(reqs), errors))
    sys.exit(1 if text_bad or req_bad else 0)


if __name__ == "__main__":
    main()
