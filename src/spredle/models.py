import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import LambdaLR
from .utils import *

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
        self.conv1 = nn.Conv1d(cfg.in_channels, cfg.out_channels, 1)
        self.conv2 = nn.Conv1d(cfg.out_channels, cfg.out_channels, 1)
        self.resblocks = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(len(cfg.NWD)):
            n,w,d = cfg.NWD[i]
            self.resblocks.append(ResidualBlock(n, w, d))
            if (i+1) % cfg.n_blocks == 0:
                self.convs.append(nn.Conv1d(cfg.out_channels, cfg.out_channels, 1))
        self.bn1 = nn.BatchNorm1d(cfg.out_channels)
        self.conv3 = nn.Conv1d(cfg.out_channels, cfg.n_classes, 1)

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

## hyena

class HyenaBlock(nn.Module):
    def __init__(self, embed_dim, max_seq_len, dropout=0.1):
        super().__init__()
        self.hyena_fwd = HyenaOperator(embed_dim, max_seq_len)
        self.hyena_bwd = HyenaOperator(embed_dim, max_seq_len)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x = x + self.attn(self.norm1(x))  # Residual + Attention
        #### bidirectional
        x = self.norm1(x)
        x_fwd = self.hyena_fwd(x)
        x_bwd = self.hyena_bwd(torch.flip(x, [1]))
        x_bwd = torch.flip(x_bwd, [1])
        x = x + (x_fwd + x_bwd)/2
        #####
        x = x + self.feed_forward(self.norm2(x))  # Residual + Feed Forward
        return x

class HyenaGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.embed_dim)
        self.blocks = nn.Sequential(*[HyenaBlock(cfg.embed_dim, cfg.max_seq_len, cfg.dropout) for _ in range(cfg.num_layers)])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = torch.nn.Conv1d(cfg.embed_dim, cfg.n_classes, 1)
        self.flank_size = cfg.flank_size

    def forward(self, x):
        B, T = x.shape  # Batch size, sequence length
        token_emb = self.token_embedding(x)  # (B, T, embed_dim)
        pos_emb = self.position_embedding(torch.arange(T, device=x.device))  # (T, embed_dim)
        x = token_emb + pos_emb.unsqueeze(0)  # Broadcast position embeddings
        x = self.blocks(x)  # Apply GPT blocks
        x = self.norm(x)  # Final LayerNorm
        x = torch.permute(x, (0, 2, 1))
        x = self.head(x)
        x = F.pad(x, [-self.flank_size, -self.flank_size])
        x = torch.softmax(x, dim=1)
        return x

