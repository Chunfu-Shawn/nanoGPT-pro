# nanoGPT-pro

Train GPT language models with PyTorch, shared RoPE, gradient accumulation, checkpoint resume, CSV and TensorBoard loss tracking, and periodic text generation.

Run the following commands from the project root. Command examples use PowerShell syntax.

**1. Set up the environment**

```powershell
conda create -n gpt python=3.11
conda activate gpt
python -m pip install torch numpy datasets tiktoken tqdm filelock tensorboard
```

**2. Download and tokenize data**

FineWeb-Edu streams from `sample-10BT` by default and prepares approximately 50 million training tokens and 500,000 validation tokens.

```powershell
python data/fineweb-edu/prepare.py
```

If `train.bin` and `val.bin` already exist, proceed to training. To prepare a smaller sample, use a separate output directory.

```powershell
python data/fineweb-edu/prepare.py --train-tokens 2000000 --val-tokens 20000 --output-dir data/fineweb-edu-small
```

Use `--dataset=fineweb-edu-small` when training on this smaller sample.

TinyStories uses the official train and validation splits. This command prepares approximately 5 million training tokens and 200,000 validation tokens. Omitting the token limits processes both complete splits.

```powershell
python data/tinystories/prepare.py --train-tokens 5000000 --val-tokens 200000
```

Both scripts use the GPT-2 tokenizer, append an end-of-text token to each document, and write `train.bin` and `val.bin` to the dataset directory. FineWeb-Edu holds out whole documents for validation. Data preparation requires access to Hugging Face and the GPT-2 tokenizer download URLs.

**3. Configure and pretrain**

Edit `config/train_nano_gpt.py`, or override settings by appending `--parameter=value` to the command.

| Parameter | Purpose |
| --- | --- |
| `dataset` | Dataset directory name, such as `fineweb-edu` or `tinystories` |
| `model_name` | Model label used in output paths |
| `n_layer`, `n_head`, `n_embd` | Number of layers, attention heads, and hidden dimensions |
| `batch_size`, `block_size` | Sequences per microbatch and tokens per sequence |
| `gradient_accumulation_steps` | Microbatches accumulated for each optimizer update |
| `learning_rate`, `warmup_ratio` | Peak learning rate and fraction of total steps used for warmup |
| `max_iters`, `lr_decay_iters` | Total optimizer steps and the step at which learning-rate decay ends |
| `rope_max_seq_len` | RoPE context limit, which must cover `block_size` |
| `eval_interval`, `generation_interval` | Validation and text-generation intervals in optimizer steps |

`n_embd` must be divisible by `n_head`, and the dimension per head must be even and greater than 2. An update typically processes `batch_size × block_size × gradient_accumulation_steps` tokens.

Start training with the configured settings.

```powershell
python train.py config/train_nano_gpt.py
```

For a shorter run, train on FineWeb-Edu for 1,000 steps, validate every 100 steps, and generate text every 500 steps.

```powershell
python train.py config/train_nano_gpt.py --dataset=fineweb-edu --max_iters=1000 --lr_decay_iters=1000 --eval_interval=100 --generation_interval=500
```

Use `device="mps"` on an Apple Silicon Mac. For CPU training, append `--device=cpu --dtype=float32 --compile=False`.

Output names include the model, dataset, learning rate, warmup ratio, and effective batch shape. An example for the current configuration is `nanoGPT-pro_fineweb-edu_lr0.001_wr0.2_batch32x512`.

```text
out/checkpoints/<run_name>.pt
out/metrics/<run_name>/steps.csv
out/metrics/<run_name>/epochs.csv
out/tensorboard/<run_name>/
```

Keep the naming parameters the same and use `--init_from=resume` to continue training. Use `--resume_checkpoint=path` to select a checkpoint explicitly.

```powershell
python train.py config/train_nano_gpt.py --init_from=resume
```

Launch TensorBoard in another terminal, then open [TensorBoard](http://localhost:6006) in your browser.

```powershell
tensorboard --logdir out/tensorboard
```

**4. Generate text**

Training generates sample text every 10,000 steps by default. Set `generation_prompt`, `generation_num_samples`, `generation_max_new_tokens`, and `generation_file` to control the prompt, sample count, output length, and destination file.

For standalone generation, set `checkpoint_path` to your saved checkpoint. This example uses the current output name and MPS.

```powershell
python sample.py --checkpoint_path=out/checkpoints/nanoGPT-pro_fineweb-edu_lr0.001_wr0.2_batch32x512.pt --device=mps --dtype=float32 --compile=False "--start=Once upon a time" --num_samples=2 --max_new_tokens=64
```

Generated text is printed to the terminal. For CPU generation, replace `--device=mps` with `--device=cpu`.
