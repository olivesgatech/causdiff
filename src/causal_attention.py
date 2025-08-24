import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalAttention(nn.Module):
    """
    Causal Attention for sub-intention sequences.
    Inputs:
      - z: (B, T, D)  sub-intention sequence (D == d_model == D_clip)
      - g: (B, D)     global goal embedding
      - key_padding_mask (optional): (B, T) with True for padded positions
    Output:
      - out: (B, T, D)

    Components:
      - IS-ATT (In-Sample): self-attention over time within the same sample (Eq.(6)).
      - CS-ATT (Cross-Sample): cross-attention from sequence to goal tokens (Eq.(7)).
        We expand goal into K learned tokens per sample to provide richer keys/values.
      - Fusion: gated fusion of IS and CS outputs, then residual + LayerNorm.

    References:
      - Front-door absorption & IS/CS sampling: Eq.(5)-(7) in the paper.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        num_goal_tokens: int = 4,   # expand G -> K tokens for richer CS attention
        dropout: float = 0.1,
        share_qkv: bool = True,     # share IS/CS projection params (recommended by paper)
        causal_mask: bool = False   # set True if you need strictly causal IS attention over time
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.num_goal_tokens = num_goal_tokens
        self.causal_mask = causal_mask

        # ---- IS-ATT (Q,K,V from the same sequence) ----
        self.q_is = nn.Linear(d_model, d_model)
        self.k_is = nn.Linear(d_model, d_model)
        self.v_is = nn.Linear(d_model, d_model)

        # ---- CS-ATT (Q from sequence; K,V from goal tokens) ----
        if share_qkv:
            # Share Q with IS, as in CATT the IS/CS share parameters to stay in same space
            self.q_cs = self.q_is
        else:
            self.q_cs = nn.Linear(d_model, d_model)

        # Project global goal G (B,D) -> goal tokens (B,K,D)
        self.goal_proj = nn.Linear(d_model, d_model * self.num_goal_tokens)
        self.k_cs = nn.Linear(d_model, d_model)
        self.v_cs = nn.Linear(d_model, d_model)

        # Output projections
        self.o_is = nn.Linear(d_model, d_model)
        self.o_cs = nn.Linear(d_model, d_model)

        # Gated fusion of IS/CS outputs
        self.fuse_gate = nn.Linear(d_model * 2, d_model)  # produces gate logits per token
        self.out_proj = nn.Linear(d_model, d_model)

        # Norms & dropout
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Simple FFN (optional but stabilizes)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def _attn(self, q, k, v, mask=None, key_padding_mask=None):
        """
        q: (B, h, Tq, Dh), k,v: (B, h, Tk, Dh)
        mask: (Tq, Tk) additive mask with -inf on disallowed
        key_padding_mask: (B, Tk) with True for pads
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)  # (B,h,Tq,Tk)

        if mask is not None:
            scores = scores + mask  # broadcast over batch/head

        if key_padding_mask is not None:
            # expand to (B,1,1,Tk)
            kpm = key_padding_mask[:, None, None, :].to(scores.dtype)
            scores = scores.masked_fill(kpm.bool(), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B,h,Tq,Dh)
        return out

    def _make_causal_mask(self, Tq, Tk, device):
        # allow j <= i; shape (Tq, Tk)
        m = torch.full((Tq, Tk), float('-inf'), device=device)
        m = torch.triu(m, diagonal=1)  # upper triangle is -inf; diagonal and below 0
        return m

    def forward(self, z, g, key_padding_mask=None):
        """
        z: (B,T,D), g: (B,D), key_padding_mask (optional): (B,T)
        """
        B, T, D = z.shape
        h, Dh = self.n_heads, self.d_head
        device = z.device
        base_dtype = z.dtype
        g = g.to(device=device, dtype=base_dtype)

        # ========= IS-ATT =========
        q_is = self.q_is(z).view(B, T, h, Dh).transpose(1, 2)  # (B,h,T,Dh)
        k_is = self.k_is(z).view(B, T, h, Dh).transpose(1, 2)  # (B,h,T,Dh)
        v_is = self.v_is(z).view(B, T, h, Dh).transpose(1, 2)  # (B,h,T,Dh)

        is_mask = None
        if self.causal_mask:
            is_mask = self._make_causal_mask(T, T, device)[None, None, :, :]  # (1,1,T,T)

        is_ctx = self._attn(q_is, k_is, v_is, mask=is_mask, key_padding_mask=key_padding_mask)
        is_ctx = is_ctx.transpose(1, 2).contiguous().view(B, T, D)           # (B,T,D)
        is_ctx = self.o_is(is_ctx)                                           # (B,T,D)

        # ========= CS-ATT =========
        # Expand global goal into K tokens per sample
        g_tokens = self.goal_proj(g).view(B, self.num_goal_tokens, D)        # (B,K,D)
        q_cs = self.q_cs(z).view(B, T, h, Dh).transpose(1, 2)                # (B,h,T,Dh)
        k_cs = self.k_cs(g_tokens).view(B, self.num_goal_tokens, h, Dh).transpose(1, 2)  # (B,h,K,Dh)
        v_cs = self.v_cs(g_tokens).view(B, self.num_goal_tokens, h, Dh).transpose(1, 2)  # (B,h,K,Dh)

        # No padding on goal tokens; mask not needed (tiny K)
        cs_ctx = self._attn(q_cs, k_cs, v_cs, mask=None, key_padding_mask=None)
        cs_ctx = cs_ctx.transpose(1, 2).contiguous().view(B, T, D)           # (B,T,D)
        cs_ctx = self.o_cs(cs_ctx)                                           # (B,T,D)

        # ========= Fuse (gated) =========
        fuse_inputs = torch.cat([is_ctx, cs_ctx], dim=-1)                    # (B,T,2D)
        gate = torch.sigmoid(self.fuse_gate(fuse_inputs))                    # (B,T,D)
        fused = gate * cs_ctx + (1.0 - gate) * is_ctx                        # (B,T,D)

        # Residual + LN
        x = self.ln1(z + self.dropout(fused))
        # FFN + Residual + LN
        x = self.ln2(x + self.ffn(x))
        # Final proj (kept identity-sized to ensure (B,T,D))
        return self.out_proj(x)
