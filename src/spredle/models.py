import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import LambdaLR

## SplcieAI

class ResidualBlock(nn.Module):
    def __init__(self, N, W, D):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(N)
        self.bn2 = nn.BatchNorm1d(N)
        self.conv1 = nn.Conv1d(N, N, W, dilation=D, padding='same')
        self.conv2 = nn.Conv1d(N, N, W, dilation=D, padding='same')
    def forward(self, x):
        out = self.bn1(x)
        out = torch.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = out + x
        return(out)

class SpliceAI(torch.nn.Module):
    '''
    nF=32
    nRB=4
    NWD1 = [[nF, 11, 1]] * nRB
    NWD2 = [[nF, 11, 1]] * nRB + [[nF, 11, 4]] * nRB
    NWD3 = [[nF, 11, 1]] * nRB + [[nF, 11, 4]] * nRB + [[nF, 21, 10]] * nRB
    NWD4 = [[nF, 11, 1]] * nRB + [[nF, 11, 4]] * nRB + [[nF, 21, 10]] * nRB + [[nF, 41, 25]] * nRB
    '''

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv1d(cfg.in_channels, cfg.n_channels, 1)
        self.conv2 = nn.Conv1d(cfg.n_channels, cfg.n_channels, 1)
        self.resblocks = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(len(cfg.NWD)):
            n,w,d = cfg.NWD[i]
            self.resblocks.append(ResidualBlock(n, w, d))
            if (i+1) % cfg.n_blocks == 0:
                self.convs.append(nn.Conv1d(cfg.n_channels, cfg.n_channels, 1))
        self.bn1 = nn.BatchNorm1d(cfg.n_channels)
        self.conv3 = nn.Conv1d(cfg.n_channels, cfg.out_channels, 1)

    def forward(self, x, from_onehot=False):
        if not from_onehot:
            x = self.OneHot(x)

        out = self.conv1(x)
        skip = self.conv2(out)
        for i in range(len(self.cfg.NWD)):
            n,w,d = self.cfg.NWD[i]
            out = self.resblocks[i](out)
            j = 0
            if (i+1)%self.cfg.n_blocks == 0:
                cv = self.convs[j](out)
                skip = cv + skip
                j += 1

        skip = nn.functional.pad(skip, [-self.cfg.flank_size, -self.cfg.flank_size])
        skip = self.bn1(skip)
        out = self.conv3(skip)
        out = torch.softmax(out, dim=1)
        return out

    def OneHot(self, x, alphabet=[1, 2, 3, 4]):
        cat = list(np.array(sorted(alphabet)).reshape(1, -1))
        oe = OneHotEncoder(categories=cat, handle_unknown='ignore')
        x = [oe.fit_transform(np.array(L).reshape(-1, 1)).toarray().transpose(1, 0) for L in x.cpu()]
        x = torch.tensor(np.array(x), dtype=torch.float32).to(self.cfg.device)
        return x

## GPT

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout
        context_vec = nn.functional.scaled_dot_product_attention(
                queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        context_vec = self.proj(context_vec)
        return context_vec

class GELU(nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
                GELU(),
                nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
                )

    def forward(self, x):
        return self.layers(x)

class LayerNorm(nn.Module):
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

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
                d_in=cfg.emb_dim,
                d_out=cfg.emb_dim,
                context_length=cfg.context_length,
                num_heads=cfg.n_heads,
                dropout=cfg.drop_rate)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x

class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_dim=32, max_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_length, emb_dim)
        k = torch.arange(0, max_length).unsqueeze(1)
        # calc divisor for positional encoding
        div_term = torch.exp(
                torch.arange(0, emb_dim, 2) * -(np.log(10000.0) / emb_dim)
                )
        # calc sine on even indices
        pe[:, 0::2] = torch.sin(k * div_term)
        # calc cosine on odd indices
        pe[:, 1::2] = torch.cos(k * div_term)
        # add dimension
        pe = pe.unsqueeze(0)
        # buffers are saved in state_dict but not trained by the optimizer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
        x:        embeddings (batch_size, context_length, emb_dim)
        Returns:
                positional encodings (batch_size, context_length, emb_dim)
        """
        x = self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        if cfg.positional_encoding:
            self.pos_emb = PositionalEncoding(cfg.emb_dim)
        else:
            self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.Sequential(
                *[TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(
                cfg.emb_dim * cfg.context_length, cfg.n_classes, bias=False
                )

    def forward(self, x):
        tok_embeds = self.tok_emb(x)

        if self.cfg.positional_encoding:
            pos_embeds = self.pos_emb(x)
        else:
            pos_embeds = self.pos_emb(torch.arange(x.shape[1]))

        x = tok_embeds + pos_embeds  # shape [batch_size, context_length, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.out_head(x)
        return logits

