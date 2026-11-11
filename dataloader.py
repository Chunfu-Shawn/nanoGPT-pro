import torch
import numpy as np

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
