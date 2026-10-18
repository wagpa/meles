import logging

import torch
from torch import nn
import torch.nn.functional as fun

from meles.frame_aggregation.iresnet import iresnet34

logger = logging.getLogger(__name__)


class FacialEncoder(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.model = iresnet34()  # Hidden state: B x 512 x 7 x 7
        if weights:
            sd = torch.load(weights, map_location="cpu")
            sd = {k.removeprefix("module."): v for k, v in sd.items()}
            self.model.load_state_dict(sd, strict=True)

    def forward(self, x):
        return self.model(x)[1]


class FacialDecoderBlock(nn.Module):
    def __init__(self, cin, cout, groups=8):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.norm1 = nn.GroupNorm(groups, cout)
        self.norm2 = nn.GroupNorm(groups, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        x = self.up(x)
        h = fun.silu(self.norm1(self.conv1(x)))
        h = fun.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class FacialDecoder(nn.Module):
    def __init__(self, cin=512, chans=(256, 192, 128, 64)):
        super().__init__()
        self.blocks = nn.ModuleList()
        for c in chans:
            self.blocks.append(FacialDecoderBlock(cin, c))
            cin = c
        self.out = nn.Sequential(nn.GroupNorm(8, cin), nn.SiLU(), nn.Conv2d(cin, 3, 3, 1, 1))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return torch.tanh(self.out(x))


class FacialAggregator(nn.Module):
    """N feature maps -> 1. One transformer block: cross-attention + FFN."""

    def __init__(self, cin=512, heads=8, grid=7, ff_mult=4, methods=0):
        """
        ``methods`` makes the pooling query conditional on which anonymization the inputs carry.
        The query is otherwise a single learned constant, identical for every input, so one model
        covering several degradations has to serve them all from one starting point -- undoing a
        blur and inventing a face behind a black bar are not the same operation. A learned
        embedding per method, added to the query, is the cheapest way to let it specialise.

        Pass 0 (the default) to keep the unconditional query, which is the right choice when one
        model is trained per anonymization.
        """
        super().__init__()
        self.grid = grid
        self.pos = nn.Parameter(torch.randn(1, grid * grid, cin) * 0.02)
        self.query = nn.Parameter(torch.randn(1, grid * grid, cin) * 0.02)
        # Zero-initialised, so an untrained conditional model starts out exactly as the
        # unconditional one does rather than with the query displaced at random.
        self.method = nn.Embedding(methods, cin) if methods else None
        if self.method is not None:
            nn.init.zeros_(self.method.weight)

        self.attn = nn.MultiheadAttention(cin, heads, batch_first=True)
        self.norm_q = nn.LayerNorm(cin)
        self.norm_kv = nn.LayerNorm(cin)
        self.norm_ff = nn.LayerNorm(cin)
        self.ff = nn.Sequential(
            nn.Linear(cin, cin * ff_mult),
            nn.GELU(),
            nn.Linear(cin * ff_mult, cin),
        )

    def forward(self, x, method=None):
        """x: B x N x C x H x W, method: B  ->  B x C x H x W"""
        b, n, c, h, w = x.shape
        t = x.flatten(0, 1).flatten(2).transpose(1, 2)  # B*N x 49 x C
        t = t + self.pos
        kv = t.reshape(b, n * h * w, c)  # B x N*49 x C
        q = self.query.expand(b, -1, -1)  # B x 49 x C
        if self.method is not None and method is not None:
            q = q + self.method(method).unsqueeze(1)  # B x 1 x C, broadcast over the grid

        h_ = q + self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv), need_weights=False)[0]
        h_ = h_ + self.ff(self.norm_ff(h_))
        return h_.transpose(1, 2).reshape(b, c, h, w)


class Facial(nn.Module):
    def __init__(self, weights=None, methods=0):
        """
        ``methods`` is how many anonymizations one model covers, i.e. ``len(CURRICULUM)`` when it
        is trained across all of them. It conditions the aggregator's pooling query on which one
        the inputs carry; 0 keeps the unconditional query. The anonymization is known at inference
        -- a sweep leg names it, and blur, blocks and a black bar are not hard to tell apart -- so
        supplying it is not information the attacker would lack.
        """
        super().__init__()
        self.encoder_frozen = False
        self.methods = methods
        self.encoder = FacialEncoder(weights=weights)
        self.aggregator = FacialAggregator(methods=methods)
        self.decoder = FacialDecoder()

    def forward(self, x, method=None):
        """x: B x N x 3 x 112 x 112, method: B -> B x 3 x 112 x 112"""
        b, n = x.shape[:2]
        if self.encoder_frozen:
            with torch.no_grad():
                f = self.encoder(x.flatten(0, 1))
        else:
            f = self.encoder(x.flatten(0, 1))
        f = f.view(b, n, *f.shape[1:])
        return self.decoder(self.aggregator(f, method))

    def train(self, mode=True):
        """Keeps the encoder in eval mode while frozen, so its BatchNorm
        running statistics are not overwritten by the new data."""
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def set_encoder_frozen(self, frozen: bool):
        self.encoder_frozen = frozen
        for p in self.encoder.parameters():
            p.requires_grad_(not frozen)
        if frozen:
            self.encoder.eval()
        return self

    def fine_parameters(self):
        """Encoder parameters: fine-tuned at a low LR. Empty while frozen."""
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def train_parameters(self):
        """Everything else: trained from scratch at the normal LR."""
        enc = {id(p) for p in self.encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in enc]
