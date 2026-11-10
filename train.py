"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import atexit
import csv
import os
import time
import math
import pickle
from contextlib import nullcontext
from tempfile import mkdtemp

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, reduce

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# metrics logging
tensorboard_log = False
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
rope_max_seq_len = 0
rope_alpha = 1.0
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_rank = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

class DataLoaderLite:
    """Read consecutive next-token batches from a flat uint16 token file."""

    def __init__(self, filename, batch_size, block_size, device, *, rank=0, world_size=1):
        self.tokens = np.memmap(filename, dtype=np.uint16, mode="r")
        self.batch_shape = (batch_size, block_size)
        self.batch_tokens = batch_size * block_size
        self.rank_offset = rank * self.batch_tokens
        self.stride = world_size * self.batch_tokens
        self.batches_per_epoch = (len(self.tokens) - 1) // self.stride
        if self.batches_per_epoch < 1:
            raise ValueError(f"{filename} needs at least {self.stride + 1:,} tokens for a full batch")
        self.device = torch.device(device)
        self.layout = (len(self.tokens), self.batch_shape, world_size)
        self.reset()

    def reset(self):
        self.batches_read = 0

    def next_batch(self):
        batch_index = self.batches_read % self.batches_per_epoch
        start = batch_index * self.stride + self.rank_offset
        tokens = self._read_tokens(start, start + self.batch_tokens + 1)
        self.batches_read += 1
        return tokens[:-1].view(self.batch_shape), tokens[1:].view(self.batch_shape)

    def _read_tokens(self, start, end):
        tokens = torch.from_numpy(self.tokens[start:end].astype(np.int64))
        use_cuda = self.device.type == "cuda"
        if use_cuda:
            tokens = tokens.pin_memory()
        return tokens.to(self.device, non_blocking=use_cuda)

    def evaluation_batches(self, limit):
        self.reset()
        count = self.batches_per_epoch if limit is None else min(limit, self.batches_per_epoch)
        for _ in range(count):
            yield self.next_batch()

        if limit is None:
            block_size = self.batch_shape[1]
            for start in range(count * self.stride, len(self.tokens) - 1, block_size):
                tokens = self._read_tokens(start, start + block_size + 1)
                yield tokens[:-1].unsqueeze(0), tokens[1:].unsqueeze(0)

    def state_dict(self):
        return {"layout": self.layout, "batches_read": self.batches_read}

    def load_state_dict(self, state):
        if state["layout"] != self.layout:
            return False
        self.batches_read = state["batches_read"]
        return True


data_dir = os.path.join("data", dataset)
train_loader = DataLoaderLite(
    os.path.join(data_dir, "train.bin"), batch_size, block_size, device,
    rank=ddp_rank, world_size=ddp_world_size,
)
if master_process:
    val_loader = DataLoaderLite(os.path.join(data_dir, "val.bin"), batch_size, block_size, device)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
epoch_loss_sum = 0.0
epoch_batches = 0

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  rope_max_seq_len=rope_max_seq_len, rope_alpha=rope_alpha,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, f'ckpt.{dataset}.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    model_args["rope_max_seq_len"] = checkpoint_model_args.get("rope_max_seq_len", 0)
    model_args["rope_alpha"] = checkpoint_model_args.get("rope_alpha", 1.0)
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    loader_state = checkpoint.get("train_loader")
    if loader_state is not None:
        if train_loader.load_state_dict(loader_state):
            epoch_loss_sum, epoch_batches = checkpoint.get("epoch_metrics", (0.0, 0))
        elif master_process:
            print("Data loader layout changed. Restarting the data pass from the beginning.")
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

def compute_loss(logits, targets):
    return F.cross_entropy(logits.flatten(0, 1), targets.reshape(-1), ignore_index=-1)


@torch.no_grad()
def estimate_loss(model, loader, limit):
    model.eval()
    loss_sum = torch.zeros((), device=device)
    token_count = 0
    for X, Y in loader.evaluation_batches(limit):
        with ctx:
            loss = compute_loss(model(X), Y)
        loss_sum += loss * Y.numel()
        token_count += Y.numel()
    model.train()
    return loss_sum.item() / token_count

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

class LossLogger:
    def __init__(self, out_dir, dataset, tensorboard_log):
        metrics_dir = os.path.join(out_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        self.directory = mkdtemp(prefix=f"{dataset}-", dir=metrics_dir)
        for interval in ("step", "epoch"):
            with open(self._csv_path(interval), "w", newline="") as handle:
                csv.writer(handle).writerow(("step", "epoch", "train_loss", "val_loss"))
        print(f"Loss CSV files: {self.directory}")

        self.writer = None
        if tensorboard_log:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                print("TensorBoard is unavailable. Install it with python -m pip install tensorboard.")
            else:
                run = os.path.basename(self.directory)
                log_dir = os.path.join(out_dir, "tensorboard", run)
                self.writer = SummaryWriter(log_dir, flush_secs=5)
                atexit.register(self.writer.close)
                print(f"TensorBoard events: {log_dir}")

    def _csv_path(self, interval):
        return os.path.join(self.directory, f"{interval}s.csv")

    def record(self, interval, step, epoch, train_loss=None, val_loss=None):
        with open(self._csv_path(interval), "a", newline="") as handle:
            csv.writer(handle).writerow((step, epoch, train_loss, val_loss))
        if self.writer is not None:
            index = step if interval == "step" else epoch
            if train_loss is not None:
                self.writer.add_scalar(f"{interval}/train_loss", train_loss, index)
            if val_loss is not None:
                self.writer.add_scalar(f"{interval}/val_loss", val_loss, index)


raw_model = model.module if ddp else model
batches_per_epoch = train_loader.batches_per_epoch
if master_process:
    loss_logger = LossLogger(out_dir, dataset, tensorboard_log)
    if wandb_log:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    if iter_num == 0 or eval_only:
        val_loss = estimate_loss(raw_model, val_loader, eval_iters)
        loss_logger.record("step", iter_num, train_loader.batches_read // batches_per_epoch,
                           val_loss=val_loss)
        print(f"step {iter_num}: val loss {val_loss:.4f}")
        best_val_loss = min(best_val_loss, val_loss)

t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
while not eval_only and iter_num < max_iters:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Finish the current data pass before starting the next optimizer step.
    remaining_batches = batches_per_epoch - train_loader.batches_read % batches_per_epoch
    micro_steps = min(gradient_accumulation_steps, remaining_batches)
    loss_sum = torch.zeros((), device=device)
    for micro_step in range(micro_steps):
        X, Y = train_loader.next_batch()
        if ddp:
            model.require_backward_grad_sync = micro_step == micro_steps - 1
        with ctx:
            loss = compute_loss(model(X), Y)
        loss_sum += loss.detach()
        scaler.scale(loss / micro_steps).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    iter_num += 1
    local_iter_num += 1

    if ddp:
        reduce(loss_sum, dst=0)

    now = time.time()
    dt = now - t0
    t0 = now
    if master_process:
        batch_loss_sum = loss_sum.item() / ddp_world_size
        step_loss = batch_loss_sum / micro_steps
        epoch_loss_sum += batch_loss_sum
        epoch_batches += micro_steps
        epoch, batch_in_epoch = divmod(train_loader.batches_read, batches_per_epoch)

        if iter_num % log_interval == 0:
            if local_iter_num > 5:
                mfu = raw_model.estimate_mfu(batch_size * micro_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {step_loss:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

        eval_step = iter_num % eval_interval == 0 or iter_num == max_iters
        val_loss = None
        save_checkpoint = always_save_checkpoint and (eval_step or batch_in_epoch == 0)
        if eval_step:
            val_loss = estimate_loss(raw_model, val_loader, eval_iters)
            print(f"step {iter_num}: train loss {step_loss:.4f}, val loss {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint = True
            if wandb_log:
                wandb.log({"iter": iter_num, "train/loss": step_loss, "val/loss": val_loss,
                           "lr": lr, "mfu": running_mfu * 100})
        loss_logger.record("step", iter_num, epoch, step_loss, val_loss)

        if batch_in_epoch == 0:
            if epoch_batches == batches_per_epoch:
                epoch_train_loss = epoch_loss_sum / epoch_batches
                epoch_val_loss = estimate_loss(raw_model, val_loader, limit=None)
                loss_logger.record("epoch", iter_num, epoch, epoch_train_loss, epoch_val_loss)
                print(f"epoch {epoch}: train loss {epoch_train_loss:.4f}, val loss {epoch_val_loss:.4f}")
            epoch_loss_sum = 0.0
            epoch_batches = 0

        if save_checkpoint:
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loader": train_loader.state_dict(),
                "epoch_metrics": (epoch_loss_sum, epoch_batches),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "config": config,
            }
            ckpt_path = os.path.join(out_dir, f"ckpt.{dataset}.pt")
            print(f"saving checkpoint to {ckpt_path}")
            torch.save(checkpoint, ckpt_path)

if ddp:
    destroy_process_group()
