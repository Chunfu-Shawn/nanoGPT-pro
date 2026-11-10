# Small TinyStories pretraining run on CPU.
out_dir = "out"
eval_interval = 100
eval_iters = 10
log_interval = 10
tensorboard_log = True
always_save_checkpoint = True

dataset = "tinystories"
gradient_accumulation_steps = 1
batch_size = 2
block_size = 128

n_layer = 6
n_head = 4
n_embd = 128
dropout = 0.1
rope_max_seq_len = 512
rope_alpha = 1.0

learning_rate = 1e-3
max_iters = 1000
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 25

device = "cpu"
dtype = "float32"
compile = False
