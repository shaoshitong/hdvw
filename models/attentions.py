from einops import rearrange
from torch import nn, einsum


class FeedForward(nn.Module):

    def __init__(self, dim_in, hidden_dim, dim_out=None, *,
                 dropout=0.0,
                 f=nn.Linear, g=nn.GELU):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out

        self.net = nn.Sequential(
            f(dim_in, hidden_dim),
            g(),
            nn.Dropout(dropout),
            f(hidden_dim, dim_out),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, head_dim=64, dropout=0.0):
        super().__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == dim)
        dim_out = dim_in if dim_out is None else dim_out

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.to_qkv = nn.Linear(dim_in, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_out),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, attn


class Transformer(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, head_dim=64, mlp_dim=1024, dropout=0.0, norm=nn.LayerNorm):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out

        self.shortcut = []
        if dim_in != dim_out:
            self.shortcut.append(norm(dim_in))
            self.shortcut.append(nn.Linear(dim_in, dim_out))
        self.shortcut = nn.Sequential(*self.shortcut)

        self.norm1 = norm(dim_in)
        self.mhsa = Attention(dim_in, dim_out, heads=heads, head_dim=head_dim, dropout=dropout)

        self.norm2 = norm(dim_out)
        self.ff = FeedForward(dim_out, mlp_dim, dim_out, dropout=dropout)

    def forward(self, x):
        skip = self.shortcut(x)
        x = self.norm1(x)
        x, attn = self.mhsa(x)
        x = x + skip

        skip = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + skip

        return x
