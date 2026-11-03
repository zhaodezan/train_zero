import argparse
import json
from pathlib import Path

from transformers import AutoConfig, AutoProcessor, AutoTokenizer, AddedToken


SOURCE = Path("Qwen/Qwen3.5-0.8B")
OUTPUT = Path("tokenizer_zh_en_math_code_16k")
VOCAB_SIZE = 16_384


def iter_corpus(rows, fields, character_budget):
    total = 0
    for row in rows:
        text = "\n".join(row[field] for field in fields if row[field])[:10_000]
        if not text.strip():
            continue
        yield text
        total += len(text)
        if total >= character_budget:
            break


def train_qwen_tokenizer(texts, source, vocab_size=VOCAB_SIZE):
    source = Path(source)
    original = AutoTokenizer.from_pretrained(source, local_files_only=True)
    added = json.loads((source / "tokenizer.json").read_text(encoding="utf-8"))["added_tokens"]
    special = [token["content"] for token in added if token["special"]]
    ordinary = [token for token in added if not token["special"]]
    tokenizer = original.train_new_from_iterator(
        texts, vocab_size=vocab_size - len(ordinary), new_special_tokens=special
    )
    tokenizer.add_tokens(
        [
            AddedToken(
                token["content"],
                single_word=token["single_word"],
                lstrip=token["lstrip"],
                rstrip=token["rstrip"],
                normalized=token["normalized"],
                special=False,
            )
            for token in ordinary
        ]
    )
    return tokenizer


def save_qwen_tokenizer(tokenizer, source, output):
    source = Path(source)
    output = Path(output)
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    config.text_config.vocab_size = len(tokenizer)
    config.text_config.eos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    for name, marker in (
        ("image_token_id", "<|image_pad|>"),
        ("video_token_id", "<|video_pad|>"),
        ("vision_start_token_id", "<|vision_start|>"),
        ("vision_end_token_id", "<|vision_end|>"),
    ):
        setattr(config, name, tokenizer.convert_tokens_to_ids(marker))
    processor.tokenizer = tokenizer
    processor.image_token_id = config.image_token_id
    processor.video_token_id = config.video_token_id
    processor.vision_start_token_id = config.vision_start_token_id
    processor.vision_end_token_id = config.vision_end_token_id
    processor.save_pretrained(output)
    config.save_pretrained(output)


def training_texts():
    from datasets import load_dataset

    sources = (
        ("wikimedia/wikipedia", "20231101.zh", ("text",), 1_000_000),
        ("wikimedia/wikipedia", "20231101.en", ("text",), 750_000),
        ("EleutherAI/hendrycks_math", "algebra", ("problem", "solution"), 200_000),
        ("openai/gsm8k", "main", ("question", "answer"), 250_000),
        ("HuggingFaceH4/CodeAlpaca_20K", None, ("prompt", "completion"), 800_000),
    )
    for name, configuration, fields, budget in sources:
        rows = load_dataset(name, configuration, split="train", streaming=True)
        count = 0
        for text in iter_corpus(rows, fields, budget):
            count += len(text)
            yield text
        print(f"{name} / {configuration}: {count:,} characters", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train a 16K Chinese-English math/code Qwen tokenizer")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output path already exists: {args.output}")
    tokenizer = train_qwen_tokenizer(training_texts(), args.source)
    save_qwen_tokenizer(tokenizer, args.source, args.output)
    print(f"Saved {len(tokenizer):,} tokens to {args.output}")


if __name__ == "__main__":
    main()
