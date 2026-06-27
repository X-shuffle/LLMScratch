"""Shared building blocks for the split GPT chapter scripts.

The original notebook relies on cells being executed in order, so variables such as
`model`, `tokenizer`, and `out` can accidentally come from a previous run. This module
collects the reusable pieces in one place so each chapter script can start from a clean
Python process and explicitly create the state it needs.
"""

from pathlib import Path
import urllib.request

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# This shorter context length is used by the evaluation/training chapters to reduce
# memory and runtime. It keeps the GPT-2-like architecture but uses 256 positions.
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

# Part 7 in the notebook originally uses GPT-2's 1024-token context length, so it
# gets a separate config to reproduce that chapter's exact random initialization.
GPT_CONFIG_124M_ORIG = {
    **GPT_CONFIG_124M,
    "context_length": 1024,
}


def get_tokenizer():
    """Return the GPT-2 BPE tokenizer used throughout the notebook."""
    return tiktoken.get_encoding("gpt2")


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention used inside each Transformer block."""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Query/key/value projections create the three views needed for attention.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # The causal mask is a non-trainable tensor stored with the module. Positions
        # above the diagonal represent "future" tokens that should be hidden.
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape

        # Project the same input sequence into keys, queries, and values.
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Split the embedding dimension into multiple heads:
        # (batch, tokens, d_out) -> (batch, tokens, heads, head_dim).
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Move heads before tokens so attention can be computed independently per head.
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Attention scores compare each query token against all key tokens.
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Scale by sqrt(head_dim) to keep softmax from becoming too peaky.
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Combine weighted values, then merge all heads back into one embedding.
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class LayerNorm(nn.Module):
    """Layer normalization with learned scale and shift parameters."""

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GELU activation used by GPT-style feed-forward networks."""

    def forward(self, x):
        return 0.5 * x * (
            1
            + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi))
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class FeedForward(nn.Module):
    """Two-layer MLP that expands then contracts the embedding dimension."""

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """One GPT Transformer block: attention, feed-forward, and residual paths."""

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Pre-norm attention block: normalize first, then add the residual shortcut.
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Pre-norm feed-forward block with the same residual pattern.
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    """Minimal GPT model assembled from token embeddings and Transformer blocks."""

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        _, seq_len = in_idx.shape
        # Token embeddings represent token identity; positional embeddings represent
        # where each token sits in the sequence.
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


def create_fresh_model(seed=123, cfg=GPT_CONFIG_124M, eval_mode=True):
    """Create a newly initialized model with a fixed seed for reproducible demos."""
    torch.manual_seed(seed)
    model = GPTModel(cfg)
    if eval_mode:
        model.eval()
    return model


def generate_text_simple(model, idx, max_new_tokens, context_size, verbose=True, top_k=3):
    """Generate text with greedy decoding.

    `idx` starts as a batch of token IDs with shape (batch, tokens). Each loop:
    crops the context, runs the model, keeps only the final time step, picks the
    highest-probability next token, and appends it to the running sequence.
    """
    for step in range(max_new_tokens):
        # Keep only the latest tokens the model can fit into its positional context.
        idx_cond = idx[:, -context_size:]
        if verbose:
            print(f"\nStep {step + 1}/{max_new_tokens}")
            print(f"  Current idx shape: {tuple(idx.shape)}")
            print(f"  Context idx_cond shape: {tuple(idx_cond.shape)}")

        with torch.no_grad():
            logits = model(idx_cond)
        if verbose:
            print(f"  Model logits shape: {tuple(logits.shape)}")

        # Only the final position predicts the next token we want to append.
        logits = logits[:, -1, :]
        if verbose:
            print(f"  Last-step logits shape: {tuple(logits.shape)}")

        # Convert raw scores over the vocabulary into probabilities for inspection.
        probas = torch.softmax(logits, dim=-1)
        if verbose:
            top_probas, top_indices = torch.topk(
                probas, k=min(top_k, probas.shape[-1]), dim=-1
            )
            for batch_idx in range(idx.shape[0]):
                candidates = [
                    f"id={token_id.item()} prob={prob.item():.4f}"
                    for token_id, prob in zip(top_indices[batch_idx], top_probas[batch_idx])
                ]
                print(f"  Batch {batch_idx} top candidates: {', '.join(candidates)}")

        # Greedy decoding: always choose the token with the largest probability.
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        if verbose:
            print(f"  Selected idx_next: {idx_next.squeeze(-1).tolist()}")

        # Append the selected token so the next loop can condition on it.
        idx = torch.cat((idx, idx_next), dim=1)
        if verbose:
            print(f"  Updated idx shape: {tuple(idx.shape)}")

    return idx


def text_to_token_ids(text, tokenizer=None):
    """Encode text and add a batch dimension: (tokens) -> (1, tokens)."""
    tokenizer = tokenizer or get_tokenizer()
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer=None):
    """Decode a batched token tensor back into a text string."""
    tokenizer = tokenizer or get_tokenizer()
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def load_verdict_text(file_path="the-verdict.txt"):
    """Load the tiny training text, downloading it once if it is missing locally."""
    path = Path(file_path)
    if not path.exists():
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        path.write_text(text_data, encoding="utf-8")
    return path.read_text(encoding="utf-8")


class GPTDatasetV1(Dataset):
    """Sliding-window dataset for next-token prediction."""

    def __init__(self, txt, tokenizer, max_length, stride, debug_name=None):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        window_starts = list(range(0, len(token_ids) - max_length, stride))

        if debug_name is not None:
            input_tokens = len(window_starts) * max_length
            unused_input_tokens = len(token_ids) - input_tokens
            print(f"[{debug_name}] total tokens: {len(token_ids)}")
            print(f"[{debug_name}] max_length: {max_length}, stride: {stride}")
            print(f"[{debug_name}] window starts: {window_starts}")
            print(f"[{debug_name}] input tokens used: {input_tokens}")
            print(f"[{debug_name}] tokens not counted as input: {unused_input_tokens}")

        # Inputs are chunks of length `max_length`; targets are shifted by one token.
        for i in window_starts:
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    debug_name=None,
):
    """Wrap GPTDatasetV1 in a PyTorch DataLoader."""
    tokenizer = get_tokenizer()
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride, debug_name=debug_name)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def create_train_val_loaders(text_data, cfg=GPT_CONFIG_124M, train_ratio=0.90):
    """Split the story into train/validation text and create matching loaders."""
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=cfg["context_length"],
        stride=cfg["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
        debug_name="train",
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=cfg["context_length"],
        stride=cfg["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
        debug_name="validation",
    )
    return train_loader, val_loader


def calc_loss_batch(input_batch, target_batch, model, device):
    """Cross-entropy loss for one batch of next-token predictions."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    # Flatten batch and time dimensions so cross_entropy sees one row per token.
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Average loss over a whole loader or over the first `num_batches` batches."""
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Temporarily switch to eval mode and compute train/validation losses."""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context, verbose=False):
    """Print a compact sample during training so progress is visible."""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
            verbose=verbose,
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    """Small educational training loop from the notebook.

    It alternates parameter updates with periodic evaluation, tracks losses, and
    prints a generated sample after each epoch.
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # Backprop computes gradients; optimizer.step applies them to weights.
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen
