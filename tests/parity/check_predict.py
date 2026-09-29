import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

FLOAT_TOL = 1e-3

REQUESTS = [
    {
        "state": "The customer says the package arrived damaged and wants a refund.",
        "questions": {
            "urgency": {
                "type": "choice",
                "instructions": "How urgent is this request?",
                "criteria": {"low": "not urgent", "medium": "somewhat urgent", "high": "very urgent"},
            },
            "satisfaction": {
                "type": "score",
                "instructions": "Rate the customer's satisfaction level.",
                "criteria": ["very unhappy", "unhappy", "neutral", "happy", "very happy"],
            },
            "needs_refund": {
                "type": "noul",
                "instructions": "Does the customer want a refund?",
            },
        },
    },
    {
        "state": "Thanks so much, this is exactly what I needed!",
        "questions": {
            "sentiment": {
                "type": "choice",
                "instructions": "What is the sentiment of this message?",
                "criteria": ["negative", "neutral", "positive"],
            },
        },
    },
    {
        "state": "Can you remind me what time the meeting is tomorrow?",
        "questions": {
            "topic": {
                "type": "choice",
                "instructions": "What is this message about?",
                "criteria": {
                    "scheduling": "meetings, calendars, appointments",
                    "billing": "payments, invoices, refunds",
                    "technical": "bugs, errors, how something works",
                    "other": "anything else",
                },
            },
            "is_question": {
                "type": "noul",
                "instructions": "Is this message asking a question?",
            },
        },
    },
    {
        "state": "I'd like to book a table for two at 7pm on Friday.",
        "questions": {
            "day": {
                "type": "choice",
                "instructions": "Which day of the week is being referenced?",
                "criteria": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
                            "Sunday", "None", "Multiple", "Unclear", "Today", "Tomorrow"],
            },
        },
    },
]


def run_binary(binary, model_dir, mode, rows, chip="cpu"):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        path = f.name
    try:
        out = subprocess.run([binary, "--model", model_dir, "--chip", chip, mode, path],
                             check=True, capture_output=True, text=True)
    finally:
        os.unlink(path)
    lines = [json.loads(line) for line in out.stdout.splitlines() if line.strip()]
    if len(lines) != len(rows):
        raise RuntimeError("binary returned %d results for %d inputs (stderr: %s)" %
                           (len(lines), len(rows), out.stderr))
    return lines


def diff_dict(want, got, path=""):
    diffs = []
    if isinstance(want, dict) and isinstance(got, dict):
        for k in want:
            if k not in got:
                diffs.append("%s: missing key %r" % (path, k))
            else:
                diffs.extend(diff_dict(want[k], got[k], path + "." + str(k)))
        for k in got:
            if k not in want:
                diffs.append("%s: unexpected key %r" % (path, k))
    elif isinstance(want, (int, float)) and isinstance(got, (int, float)) and not isinstance(want, bool):
        if abs(float(want) - float(got)) > FLOAT_TOL:
            diffs.append("%s: want %r got %r" % (path, want, got))
    elif want != got:
        diffs.append("%s: want %r got %r" % (path, want, got))
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(REPO_ROOT, "models", "laya"))
    ap.add_argument("--binary", default=os.path.join(REPO_ROOT, "build", "laya-predict"))
    ap.add_argument("--chip", default="cpu")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    from laya.agent import Agent

    agent = Agent(args.model, device="cpu")

    mine_results = run_binary(args.binary, args.model, "requests", REQUESTS, chip=args.chip)

    matched = 0
    for req, mine in zip(REQUESTS, mine_results):
        [want] = agent.predict_batch([req["state"]], req["questions"])
        diffs = diff_dict(want, mine)
        if diffs:
            print("MISMATCH for state=%r" % (req["state"][:50],))
            for d in diffs:
                print("  " + d)
        else:
            matched += 1
            if args.verbose:
                print("match: %r -> %s" % (req["state"][:50], json.dumps(mine)))

    print("model: %s" % args.model)
    print("requests: %d/%d match" % (matched, len(REQUESTS)))
    sys.exit(1 if matched != len(REQUESTS) else 0)


if __name__ == "__main__":
    main()
