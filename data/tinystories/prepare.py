import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tiktoken
from datasets import load_dataset


def write_tokens(documents, encoder, filename, target_tokens=None):
    token_count = 0
    with filename.open("wb") as output:
        for document in documents:
            text = document["text"]
            if not text.strip():
                continue
            ids = encoder.encode_ordinary(text)
            ids.append(encoder.eot_token)
            np.asarray(ids, dtype=np.uint16).tofile(output)
            token_count += len(ids)
            if target_tokens is not None and token_count >= target_tokens:
                return token_count
            
        return token_count

    raise ValueError(f"{str(filename)}: expected {target_tokens:,} tokens, got {token_count:,}")


def main():
    parser = argparse.ArgumentParser(description="Prepare the official TinyStories splits for nanoGPT.")
    parser.add_argument("--train-tokens", type=int, default=None)
    parser.add_argument("--val-tokens", type=int, default=None)
    args = parser.parse_args()
    if args.train_tokens is not None and args.train_tokens <= 0:
        parser.error("train token target must be positive")
    if args.val_tokens is not None and args.val_tokens <= 0:
        parser.error("val token target must be positive")

    output_dir = Path(__file__).resolve().parent
    train_bin = output_dir / "train.bin"
    val_bin = output_dir / "val.bin"
    # Load encoder once and reuse for all splits
    encoder = tiktoken.get_encoding("gpt2")
    dataset = load_dataset("roneneldan/TinyStories", streaming=True)
    train = dataset["train"].shuffle(seed=42, buffer_size=2048)
    val = dataset["validation"]

    train_count = write_tokens(train, encoder, train_bin, args.train_tokens)
    val_count = write_tokens(val, encoder, val_bin, args.val_tokens)

    print(f"train.bin: {train_count:,} tokens")
    print(f"val.bin:   {val_count:,} tokens")
    print(f"Output:    {output_dir}")


if __name__ == "__main__":
    main()
