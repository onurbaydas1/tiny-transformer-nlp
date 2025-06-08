import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_size, heads)
        self.ln1 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.ReLU(),
            nn.Linear(emb_size * 4, emb_size)
        )
        self.ln2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        return self.ln2(x + ff_out)

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, emb_size=64, num_heads=2, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_size)
        self.blocks = nn.Sequential(*[TransformerBlock(emb_size, num_heads) for _ in range(num_layers)])
        self.fc = nn.Linear(emb_size, vocab_size)

    def forward(self, x):
        x = self.embed(x).transpose(0, 1)  # (seq_len, batch, emb)
        x = self.blocks(x)
        x = self.fc(x).transpose(0, 1)
        return x
