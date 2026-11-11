import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tiktoken
from datasets import load_dataset
from filelock import FileLock
from tqdm.auto import tqdm


def write_tokens(documents, encoder, filename, target_tokens=None):
    token_count = 0
    with filename.open("wb") as output, tqdm(
        total=target_tokens, desc=filename.stem, unit="token", unit_scale=True,
    ) as progress:
        for document in documents:
            text = document["text"]
            if not text.strip():
                continue
            ids = encoder.encode_ordinary(text)
            ids.append(encoder.eot_token)
            np.asarray(ids, dtype=np.uint16).tofile(output)
            token_count += len(ids)
            progress.update(len(ids))
            if target_tokens is not None and token_count >= target_tokens:
                return token_count

    if token_count == 0 or (target_tokens is not None and token_count < target_tokens):
        raise ValueError(f"{filename.name}: expected {target_tokens} tokens, got {token_count:,}")
    return token_count


def main():
    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu for nanoGPT.")
    parser.add_argument("--subset", default="sample-10BT", help="FineWeb-Edu dataset configuration.")
    parser.add_argument("--train-tokens", type=int, default=50_000_000)
    parser.add_argument("--val-tokens", type=int, default=500_000)
    parser.add_argument("--full", action="store_true", help="Read the entire selected subset after reserving validation data.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.val_tokens <= 0 or (not args.full and args.train_tokens <= 0):
        parser.error("token targets must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_bin, val_bin = output_dir / "train.bin", output_dir / "val.bin"
    with FileLock(output_dir / ".prepare.lock", timeout=0):
        if train_bin.exists() or val_bin.exists():
            parser.error(f"Token files already exist in {output_dir}. Choose another --output-dir.")

        encoder = tiktoken.get_encoding("gpt2")
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu", name=args.subset, split="train", streaming=True,
        )
        documents = iter(dataset.shuffle(seed=42, buffer_size=2048))
        train_target = None if args.full else args.train_tokens
        with TemporaryDirectory(dir=output_dir) as temporary_dir:
            staging = Path(temporary_dir)
            # Consume each document once so the two splits use separate rows.
            val_count = write_tokens(documents, encoder, staging / "val.bin", args.val_tokens)
            train_count = write_tokens(documents, encoder, staging / "train.bin", train_target)
            os.link(staging / "train.bin", train_bin)
            try:
                os.link(staging / "val.bin", val_bin)
            except BaseException:
                train_bin.unlink()
                raise

    print(f"Subset:    {args.subset}")
    print(f"train.bin: {train_count:,} tokens")
    print(f"val.bin:   {val_count:,} tokens")
    print(f"Output:    {output_dir}")


if __name__ == "__main__":
    main()
