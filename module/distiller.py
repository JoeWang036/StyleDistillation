from torch import nn, einsum
from einops import rearrange
import torch


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        h = self.heads

        q = self.to_q(x)
        context = context
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class DistillationModule_v2(torch.nn.Module):
    def __init__(self, dim_t, dim_i, n_heads, d_head, style_emb, content_emb, image_emb, image_emb_patch, dropout=0.):
        super().__init__()
        self.content_emb = nn.Parameter(content_emb)
        self.style_emb = nn.Parameter(style_emb)
        self.image_emb = nn.Parameter(image_emb)
        self.image_emb_patch = nn.Parameter(image_emb_patch)
        self.content_emb.requires_grad = False
        self.style_emb.requires_grad = False
        self.image_emb.requires_grad = False
        self.image_emb_patch.requires_grad = False

        self.image_patch_proj = nn.Linear(image_emb_patch.shape[-1], dim_i)

        self.attn_c = CrossAttention(query_dim=dim_t, context_dim=dim_i,
                                     heads=n_heads, dim_head=d_head, dropout=dropout)
        self.attn_s = CrossAttention(query_dim=dim_t, context_dim=dim_i,
                                     heads=n_heads, dim_head=d_head, dropout=dropout)
        self.net_c = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_t, dim_t))
        self.net_s = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_t, dim_t))

    def forward(self):
        c = self.content_emb
        s = self.style_emb
        i = self.image_emb
        i_patch = self.image_emb_patch
        c_input = c.unsqueeze(1)
        s_input = s.unsqueeze(1)
        i_input = self.image_patch_proj(i_patch)

        c_output = self.attn_c(c_input, i_input)
        c_output = self.net_c(c_output)

        s_output = self.attn_s(s_input, i_input)
        s_output = self.net_s(s_output)

        c_output = c_output.squeeze(1)
        s_output = s_output.squeeze(1)

        return s_output, c_output
