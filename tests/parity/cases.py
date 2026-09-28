import json
import random

TEXTS = [
    "",
    " ",
    "  ",
    "Hello world",
    " Hello world",
    "Hello  world",
    "Hello   world   ",
    "trailing spaces   ",
    "\n\nleading newlines",
    "tabs\tand\ttabs\t\t",
    "mixed \t\n whitespace \r\n end",
    "it's we're they've I'm you'll he'd don't",
    "IT'S WE'RE 'S 'T 'RE",
    "''s '' ' 'x",
    "numbers 123 4567890 3.14159 -42 1e10 ½ ² ٣",
    "punctuation!!! ??? ... --- (parens) [brackets] {braces} <angles>",
    "email me at john.doe@example.com or call +1 (555) 012-3456",
    "https://example.com/path?q=1&r=two#frag",
    "café naïve résumé Ünïcödé",
    "café decomposed é Å",
    "Å angstrom sign and Ω ohm sign",
    "東京は晴れです。日本語のテキスト",
    "中文文本没有空格分隔的长句子测试",
    "한국어 텍스트 입니다",
    "Привет, мир! Ελληνικά κείμενο",
    "مرحبا بالعالم עברית",
    "हिन्दी पाठ ภาษาไทย",
    "emoji 😀👍🏽👨‍👩‍👧‍👦 🇺🇸 ❤️",
    "zero​width‍joiner﻿bom",
    "nbsp space ideographic　space line sep para sep",
    "control\x00chars\x01\x1f\x7f",
    "[MASK]",
    "a [MASK] b",
    "a    [MASK] b",
    "[MASK][MASK] [MASK]",
    "[CLS] literal [SEP] tokens [PAD] and [UNK]",
    "[mask] lowercase is not special",
    "|||EMAIL_ADDRESS||| redacted",
    "<|endoftext|>",
    "<|endoftext|> after",
    "code:\n    def f(x):\n        return x * 2  # double\n",
    '{"user": "hi", "turns": [{"role": "assistant", "content": "hello"}]}',
    "a" * 300,
    "ab" * 200,
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
    "The quick brown fox jumps over the lazy dog. " * 20,
]

ALPHABETS = [
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "0123456789",
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
    " ",
    "   \t\n\r",
    " 　 \u0085\u000b\u000c  ",
    "éüñçøåßœæÆÉ",
    "̧́̈̊",
    "日本語中文字漢",
    "ひらがなカタカナ",
    "한국어",
    "Привет",
    "مرحبا",
    "😀👍🏽❤️🇺🇸‍️",
    "½²٣⅓Ⅷ",
    "'s'll're've'd't'm",
]

SPECIALS = ["[MASK]", "[CLS]", "[SEP]", "[PAD]", "[UNK]", "|||EMAIL_ADDRESS|||", "<|endoftext|>"]


def random_text(rng, max_len=80):
    out = []
    n = rng.randint(0, max_len)
    while len(out) < n:
        r = rng.random()
        if r < 0.03:
            out.append(rng.choice(SPECIALS))
        elif r < 0.1:
            out.append(" " * rng.randint(2, 30))
        else:
            alpha = rng.choice(ALPHABETS)
            out.extend(rng.choice(alpha) for _ in range(rng.randint(1, 8)))
    return "".join(out)


BASE_QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"billing": "payments, refunds, invoices", "support": "", "sales": None},
    },
    "labels": {"type": "choice", "instructions": "Pick one", "criteria": ["yes", "no", "maybe"]},
    "scalar_labels": {"type": "choice", "instructions": "Pick", "criteria": [1, 2.5, False, "x", 1e-07]},
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["not urgent", "somewhat urgent", {"desc": "critical", "sla": 1}],
    },
    "angry": {"type": "noul", "instructions": "Is the customer angry?"},
    "angry2": {
        "type": "noul",
        "instructions": "Is the customer angry?",
        "criteria": {"TRUE": "they are shouting", "false": ""},
        "labels": {"false": "  calm ", "true": "angry"},
    },
    "structured_ins": {"type": "noul", "instructions": {"rule": "café must be open", "n": 3}},
    "masked": {
        "type": "choice",
        "instructions": "Text with [MASK] inside the question [MASK]",
        "criteria": {"a [MASK] b": "desc with [MASK]", "c": 0, "d": False},
    },
}

STATES = [
    "I was charged twice for my order and nobody answers my emails!!!",
    "",
    "short",
    {"ticket": "refund please", "amount": 42.5, "tiny": 1.5e-05, "big": 1e16, "whole": 3.0, "neg": -0.0,
     "flag": True, "none": None, "nested": {"é": ["ü", 1, 2]}, "ctrl": "line\nbreak\ttab\"quote\\"},
    [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
     {"role": "user", "content": "cancel my subscription now"}],
    "a [MASK] in the state [MASK]",
    "The quick brown fox jumps over the lazy dog. " * 120,
    ["turn %d says something different" % i for i in range(200)],
    12345,
    "東京は晴れです。" * 50,
]


def request_cases():
    cases = []
    for state in STATES:
        cases.append({"state": state, "questions": BASE_QUESTIONS})
    many = {"type": "choice", "instructions": "Classify the intent",
            "criteria": ["intent_%02d_%s" % (i, "x" * (i % 7)) for i in range(77)]}
    cases.append({"state": STATES[0], "questions": {"banking77": many}})
    shared = {"type": "choice", "instructions": "Pick",
              "criteria": ["the same long shared prefix for every option number %d" % i for i in range(40)]}
    cases.append({"state": STATES[0], "questions": {"collide": shared}})
    long_opts = {"type": "choice", "instructions": "long " * 300,
                 "criteria": {"k%d" % i: "description " * 60 for i in range(5)}}
    cases.append({"state": STATES[6], "questions": {"long": long_opts}})
    cases.append({"state": STATES[6], "questions": BASE_QUESTIONS, "max_len": 64, "head_max_len": 32})
    cases.append({"state": STATES[7], "questions": BASE_QUESTIONS, "max_len": 128})
    cases.append({"state": STATES[0], "questions": {"route": BASE_QUESTIONS["route"]}, "max_len": 12, "head_max_len": 8})
    cases.append({"state": STATES[0], "questions": {}})
    cases.append({"state": STATES[0], "questions": {"bad": {"type": "rank", "instructions": "x"}}})
    cases.append({"state": STATES[0], "questions": {"bad": {"type": "choice", "instructions": "x", "criteria": []}}})
    cases.append({"state": STATES[0], "questions": {"bad": {"type": "choice", "instructions": "x", "criteria": [1, 1.0]}}})
    cases.append({"state": STATES[0], "questions": {"bad": {"type": "score", "instructions": "x", "criteria": ["a", None]}}})
    cases.append({"state": STATES[0], "questions": {"bad": {"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}}}})
    cases.append({"state": STATES[0], "questions": {"bad": {"type": "noul", "instructions": "x", "labels": {"false": "a", "true": " a "}}}})
    cases.append({"state": STATES[0], "questions": {"bad": {"type": "choice", "instructions": "x", "criteria": ["a"], "labels": {}}}})
    cases.append({"state": STATES[0], "questions": {"bad": {"type": "choice", "criteria": ["a"]}}})
    return cases


def random_request(rng):
    def rand_value(depth=0):
        r = rng.random()
        if depth < 2 and r < 0.15:
            return {random_text(rng, 8) or "k": rand_value(depth + 1) for _ in range(rng.randint(0, 4))}
        if depth < 2 and r < 0.3:
            return [rand_value(depth + 1) for _ in range(rng.randint(0, 4))]
        if r < 0.4:
            return rng.choice([rng.randint(-10**6, 10**6), rng.uniform(-1e3, 1e3), rng.random() * 10 ** rng.randint(-8, 20)])
        if r < 0.45:
            return rng.choice([True, False, None])
        return random_text(rng, 40)

    questions = {}
    for i in range(rng.randint(1, 4)):
        kind = rng.choice(["choice", "choice_list", "score", "noul"])
        ins = random_text(rng, 60) if rng.random() < 0.8 else rand_value()
        if kind == "choice":
            crit = {}
            for _ in range(rng.randint(1, 12)):
                crit[random_text(rng, 10) or "opt"] = rng.choice([None, "", random_text(rng, 80), rand_value()])
            q = {"type": "choice", "instructions": ins, "criteria": crit}
        elif kind == "choice_list":
            labels = list(dict.fromkeys(random_text(rng, 10) or "opt" for _ in range(rng.randint(1, 30))))
            q = {"type": "choice", "instructions": ins, "criteria": labels}
        elif kind == "score":
            q = {"type": "score", "instructions": ins,
                 "criteria": [rng.choice([random_text(rng, 50), rand_value()]) or "lvl" for _ in range(rng.randint(1, 6))]}
        else:
            q = {"type": "noul", "instructions": ins}
            if rng.random() < 0.5:
                q["criteria"] = {k: random_text(rng, 40) for k in rng.sample(["true", "false"], rng.randint(1, 2))}
        questions["q%d" % i] = q
    state = rng.choice([random_text(rng, 400), rand_value(), [random_text(rng, 50) for _ in range(rng.randint(0, 30))]])
    if state is None:
        state = ""
    req = {"state": state, "questions": questions}
    if rng.random() < 0.3:
        req["max_len"] = rng.choice([16, 32, 64, 128, 256])
        req["head_max_len"] = rng.choice([8, 16, 48, 96])
    return json.loads(json.dumps(req, ensure_ascii=False))


def fuzz_texts(seed, n):
    rng = random.Random(seed)
    return [random_text(rng) for _ in range(n)]


def fuzz_requests(seed, n):
    rng = random.Random(seed + 1)
    return [random_request(rng) for _ in range(n)]
