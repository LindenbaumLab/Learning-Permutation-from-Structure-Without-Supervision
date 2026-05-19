import torch.nn as nn

class FCModel(nn.Module):
    def __init__(self, hid_c, out_c):
        super().__init__()

        self.g1 = nn.Sequential(
            nn.Conv1d(1, hid_c, 1),
            nn.ReLU(True)
        )

        self.g2 = nn.Sequential(
            nn.Conv1d(hid_c, out_c, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, attn_bias=None):
        # NOTE: x is 2-dim or 3-dim torch.Tensor, i.e., (batch, 1, numbers)
        if x.dim() == 2:
            x = x[:, None]
        h = self.g1(x)
        log_alpha = self.g2(h).transpose(1,2).contiguous()
        return log_alpha


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(True),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_bias, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class TransformerEncoderModel(nn.Module):
    def __init__(self, d_model, out_c, n_layers=8, n_heads=4, ffn_dim=128, dropout=0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.input_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)]
        )
        self.out_proj = nn.Linear(d_model, out_c)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _expand_attn_bias(self, attn_bias):
        if attn_bias is None:
            return None
        if attn_bias.dim() != 3:
            raise ValueError("attn_bias must be (batch, n, n)")
        return attn_bias.repeat_interleave(self.n_heads, dim=0)

    def forward(self, x, attn_bias=None):
        # Accept (B, n), (B, 1, n), or (B, n, 1).
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 3 and x.size(1) == 1:
            x = x.transpose(1, 2).contiguous()
        h = self.input_proj(x)
        h = h.transpose(0, 1).contiguous()
        expanded_bias = self._expand_attn_bias(attn_bias)
        for block in self.blocks:
            h = block(h, attn_bias=expanded_bias)
        h = h.transpose(0, 1).contiguous()
        log_alpha = self.out_proj(h)
        return log_alpha


class ThresholdPolicy(nn.Module):
    def __init__(self, n_numbers, hid_c):
        super().__init__()
        in_dim = n_numbers + 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_c),
            nn.ReLU(True),
            nn.Linear(hid_c, hid_c),
            nn.ReLU(True),
        )
        self.mean_head = nn.Linear(hid_c, 1)
        self.logstd_head = nn.Linear(hid_c, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        h = self.net(features)
        mean = self.mean_head(h)
        logstd = self.logstd_head(h).clamp(-5.0, 2.0)
        return mean, logstd
