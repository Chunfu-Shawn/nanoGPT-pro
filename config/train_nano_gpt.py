# Small TinyStories pretraining run on CPU.
out_dir = "out"
eval_interval = 10000
eval_iters = 500
log_interval = 200
tensorboard_log = True
always_save_checkpoint = True

dataset = "fineweb-edu" # "tinystories"
gradient_accumulation_steps = 2
batch_size = 16
block_size = 512

model_name='nanoGPT-pro'
n_layer = 12
n_head = 8
n_embd = 256
dropout = 0.1
rope_max_seq_len = 1024
rope_alpha = 1.0

# training
init_from="scratch"
learning_rate = 1e-3
max_iters = 500000
lr_decay_iters = 400000
min_lr = 1e-4
beta2 = 0.99
warmup_ratio = 0.2

generation_interval=10000
generation_file="out/fineweb-edu_samples.jsonl"
generation_prompt = "Hello, I'm a language model,"
generation_num_samples = 4
generation_max_new_tokens = 64

device = "mps"
dtype = "float16"
compile = True
