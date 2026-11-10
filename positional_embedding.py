import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, p_drop=None, max_seq_length=20000):
        super().__init__()
        self.dropout = nn.Dropout(p=p_drop) if p_drop is not None else None
        position_id = torch.arange(0, max_seq_length).unsqueeze(1)  # (max_seq_length, 1)
        frequencies = torch.pow(10000., -torch.arange(0, d_model, 2, dtype=torch.float) / d_model)

        pe = torch.zeros(max_seq_length, d_model)
        # sine on even positions
        pe[:, 0::2] = torch.sin(position_id * frequencies)
        # cosine on odd positions (different dimensions)
        pe[:, 1::2] = torch.cos(position_id * frequencies) if d_model % 2 == 0 else torch.cos(position_id * frequencies)[:,:-1]
        self.register_buffer('pe', pe)

    def forward(self, embeddings_batch):
        # embedding_batch  shape: (bs, seq_len, d_model)
        # pe shape: (max_seq_length, d_model)
        # pe shape broad_casted -> (bs, seq_len, d_model)
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.pe.shape[-1]
        positional_encodings = embeddings_batch + self.pe[:embeddings_batch.shape[1]]
        if self.dropout is not None:
            positional_encodings = self.dropout(positional_encodings)
        return positional_encodings
    

class ExtendedRotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=30000, base=10000, alpha=8, device=None):
        super().__init__()
        if dim <= 2 or dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even and greater than 2")
        base = base * alpha ** (dim / (dim - 2))
        # two embedding dimension for one group for rotation theta_i
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        # token index t
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # freqs shape = [seq_len, dim//2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # torch.outer, calculate m*theta 
        # cache shape: [1, 1, seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
    
    def rotate_half(self, x):
        # split the last dimension in half, then rotate via complex multiplication by i
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, positions=None):
        """
        Apply RoPE to x using absolute positions.

        x: (bs, n_heads, seq_len, head_dim)
           note: seq_len may be 1 for single-step decode
        positions:
           - None: fallback to original behavior using 0..seq_len-1 (as before).
           - int: treat as starting absolute position => use positions start..start+seq_len-1 for all batch entries
           - tensor (bs,): per-batch absolute positions for the first token in x (useful for seq_len==1)
           - tensor (bs, seq_len): explicit positions for each element (advanced)
        """
        batch_size, _, seq_len, head_dim = x.shape
        if head_dim != self.cos_cached.size(-1):
            raise ValueError("Input head dimension does not match the RoPE cache")

        if positions is None or isinstance(positions, int):
            start = 0 if positions is None else positions
            end = start + seq_len
            if start < 0 or end > self.max_seq_len_cached:
                raise ValueError("Position range exceeds the RoPE cache")

            cos = self.cos_cached[:, :, start:end, :]
            sin = self.sin_cached[:, :, start:end, :]
        else:
            ids = torch.as_tensor(positions, device=self.cos_cached.device)
            if ids.is_floating_point() or ids.is_complex() or ids.dtype == torch.bool:
                raise TypeError("Position indices must be integers")
            ids = ids.long()

            if ids.shape == (batch_size,):
                offsets = torch.arange(seq_len, device=ids.device) # offsets.shape = [seq_len]
                ids = ids[:, None] + offsets # ids.shape = [bs, seq_len]
            elif ids.shape != (batch_size, seq_len):
                raise ValueError("Expected positions with shape (B,) or (B, T)")

            if ((ids < 0) | (ids >= self.max_seq_len_cached)).any():
                raise ValueError("Position indices exceed the RoPE cache")

            cos = self.cos_cached[0, 0, ids].unsqueeze(1)
            sin = self.sin_cached[0, 0, ids].unsqueeze(1)

        return x * cos.to(x) + self.rotate_half(x) * sin.to(x)
