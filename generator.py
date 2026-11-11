import torch


def generate_text(model, prompt, *, meta=None, num_samples=4, max_new_tokens=64,
                  temperature=0.8, top_k=50, seed=42):
    """Generate and decode continuations without advancing the training RNG."""
    if num_samples < 1 or max_new_tokens < 1 or temperature <= 0:
        raise ValueError("Generation requires positive sample count, token count, and temperature")
    if meta is not None and "stoi" in meta and "itos" in meta:
        try:
            prompt_ids = [meta["stoi"][char] for char in prompt]
        except KeyError as exc:
            raise ValueError(f"Prompt contains a character outside the vocabulary: {exc}") from exc
        vocab_size = meta["vocab_size"]
        eos_id = None

        def decode(ids):
            return "".join(meta["itos"][token] for token in ids)
    else:
        import tiktoken

        encoding = tiktoken.get_encoding("gpt2")
        prompt_ids = encoding.encode(prompt, allowed_special={"<|endoftext|>"})
        vocab_size = encoding.n_vocab
        eos_id = encoding.eot_token
        decode = encoding.decode

    if not prompt_ids:
        raise ValueError("Generation prompt must contain at least one token")

    device = next(model.parameters()).device
    tokens = torch.tensor(prompt_ids, dtype=torch.long, device=device).repeat(num_samples, 1)
    sample_rng = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        generated = model.generate(
            tokens, max_new_tokens, temperature, top_k,
            vocab_size=vocab_size, sample_rng=sample_rng,
        )
    finally:
        model.train(was_training)

    completions = []
    for ids in generated[:, len(prompt_ids):].tolist():
        if eos_id in ids:
            ids = ids[:ids.index(eos_id)]
        completions.append(decode(ids))
    return completions
