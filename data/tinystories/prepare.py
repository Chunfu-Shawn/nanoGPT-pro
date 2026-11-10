import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tiktoken
from datasets import load_dataset


def write_tokens(documents, encoder, filename, target_tokens):
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
            if token_count >= target_tokens:
                return token_count

    raise ValueError(f"{filename.name}: expected {target_tokens:,} tokens, got {token_count:,}")


def main():
    parser = argparse.ArgumentParser(description="Prepare the official TinyStories splits for nanoGPT.")
    parser.add_argument("--train-tokens", type=int, default=5_000_000)
    parser.add_argument("--val-tokens", type=int, default=200_000)
    args = parser.parse_args()
    if min(args.train_tokens, args.val_tokens) <= 0:
        parser.error("token targets must be positive")

    output_dir = Path(__file__).resolve().with_name("tinystories")
    if output_dir.exists():
        parser.error(f"{output_dir} already exists. Move it before preparing a new sample.")

    encoder = tiktoken.get_encoding("gpt2")
    dataset = load_dataset("roneneldan/TinyStories", streaming=True)
    train = dataset["train"].shuffle(seed=42, buffer_size=2048)
    val = dataset["validation"]

    with TemporaryDirectory(dir=output_dir.parent) as temporary_dir:
        prepared = Path(temporary_dir) / "tinystories"
        prepared.mkdir()
        train_count = write_tokens(train, encoder, prepared / "train.bin", args.train_tokens)
        val_count = write_tokens(val, encoder, prepared / "val.bin", args.val_tokens)
        prepared.rename(output_dir)

    print(f"train.bin: {train_count:,} tokens")
    print(f"val.bin:   {val_count:,} tokens")
    print(f"Output:    {output_dir}")


if __name__ == "__main__":
    main()
