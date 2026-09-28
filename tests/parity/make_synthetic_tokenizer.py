import argparse
import json
import os

from tokenizers import AddedToken, Tokenizer, models, normalizers, pre_tokenizers, trainers

CORPUS = [
    "The quick brown fox jumps over the lazy dog. It's 2026 and we're shipping v3.14 builds!",
    "Refund request: order #A-99812 was charged twice ($42.50). Customer's email is a@b.co.",
    "choice question: Which team should handle this ticket? [MASK] billing [MASK] support",
    "noul question: Is the user angry? false: no, the statement does not hold true: yes",
    "score question: Rate urgency level 0: none level 1: low level 2: high",
    "Ünïcödé naïve café résumé — “quotes” ‘single’ … 東京 は 晴れ です 😀👍🏽 🇺🇸",
    "Tabs\tand\nnewlines\r\n    indented code:\n        return x  # comment",
    '{"user": "hi", "turns": [{"role": "assistant", "content": "hello"}], "n": 1.5e-05}',
    "don't can't won't I'll you've they'd she's I'm DON'T",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "tokenizer"), exist_ok=True)

    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    trainer = trainers.BpeTrainer(
        vocab_size=1200,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(CORPUS * 20, trainer=trainer)

    tok.add_tokens([AddedToken("|||EMAIL_ADDRESS|||", normalized=True)])
    tok.add_tokens([AddedToken(" " * n, normalized=True) for n in range(2, 25)])
    tok.add_special_tokens([
        AddedToken("[UNK]", normalized=False),
        AddedToken("[CLS]", normalized=False),
        AddedToken("[SEP]", normalized=False),
        AddedToken("[PAD]", normalized=False),
        AddedToken("[MASK]", lstrip=True, normalized=False),
        AddedToken("<|endoftext|>", normalized=False),
    ])
    tok.save(os.path.join(args.out, "tokenizer", "tokenizer.json"))

    with open(os.path.join(args.out, "tokenizer", "tokenizer_config.json"), "w") as f:
        json.dump({
            "tokenizer_class": "PreTrainedTokenizerFast",
            "cls_token": "[CLS]", "sep_token": "[SEP]", "pad_token": "[PAD]",
            "mask_token": "[MASK]", "unk_token": "[UNK]",
            "model_max_length": 8192,
        }, f, indent=2)
    with open(os.path.join(args.out, "rl_agent_config.json"), "w") as f:
        json.dump({"max_len": 512, "head_max_len": 192}, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
