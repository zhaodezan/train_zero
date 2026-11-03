from pathlib import Path

from transformers import AutoTokenizer


TOKENIZER_DIR = Path(__file__).resolve().parent / "tokenizer_zh_en_math_code_16k"


def print_token_details(tokenizer, token_ids):
    print("ID     词表写法           单独解码")
    for token_id in token_ids:
        raw = tokenizer.convert_ids_to_tokens(token_id)
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        display = "字节片段，需与相邻 token 合并" if "�" in decoded else repr(decoded)
        print(f"{token_id:<6} {raw:<18} {display}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    print(f"词表大小: {len(tokenizer)}")

    samples = (
        "你好，Hello!",
        r"设 $x^2 + y^2 = 1$，计算 \frac{x}{y}",
        "def square(x):\n    return x ** 2  # 平方\n",
        "<|im_start|>user\n你好<|im_end|>",
    )
    for text in samples:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        print(f"\n原文: {text!r}")
        print(f"Token IDs: {token_ids}")
        print_token_details(tokenizer, token_ids)
        print(f"解码: {decoded!r}")
        print(f"往返一致: {decoded == text}")


if __name__ == "__main__":
    main()
