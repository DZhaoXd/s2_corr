# --------------------------------------------------------
# Mamba-lite with chunk-based State Passing
# --------------------------------------------------------

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM2D(nn.Module):
    def __init__(self, in_app, c):
        super().__init__()
        if in_app > 0:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Conv2d(in_app, c * 2, 1, bias=True)
        else:
            self.pool = None
            self.fc = None

    def forward(self, x, app):  # x:[B,C,H,W], app:[B,Ca,H,W] or None
        if self.fc is None or app is None:
            return x
        s = self.fc(self.pool(app))  # [B,2C,1,1]
        gamma, beta = torch.chunk(s, 2, dim=1)
        return x * torch.sigmoid(gamma) + beta

class Mamba1DWithDecayPrior(nn.Module):
    """
    1D SSM scan + head-wise gamma prior:
      - forward / reverse bidirectional scanning
      - supports cross-chunk state passing via init_state_fwd / init_state_rev
    """
    def __init__(self, d_model, num_heads=4, gamma_range=(0.6, 0.98),
                 fuse_mode='gate', dropout=0.0, bidi=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.d = d_model
        self.h = num_heads
        self.dh = d_model // num_heads
        self.bidi = bidi
        self.fuse_mode = fuse_mode

        self.out_norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())
        self.proj_in = nn.Linear(d_model, 3 * d_model)      # beta, gamma, delta
        self.alpha_logit = nn.Parameter(torch.zeros(d_model))
        self.mix_w = nn.Parameter(torch.zeros(d_model))
        self.drop = nn.Dropout(dropout)

        lo, hi = gamma_range
        gammas = torch.linspace(lo, hi, steps=self.h)
        self.register_buffer("gamma_heads", gammas, persistent=False)

        if self.bidi and self.fuse_mode == 'gate':
            self.fuse_gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model)
            )

    def _alpha_eff(self):
        sig = torch.sigmoid
        alpha_l = sig(self.alpha_logit)                  # learnable α
        w = sig(self.mix_w)                              # mixing weight
        g = self.gamma_heads.repeat_interleave(self.dh)  # head-wise γ prior
        alpha_eff = w * alpha_l + (1 - w) * g
        return alpha_eff.view(1, 1, self.d)

    def _scan_1d(self, x, alpha_eff, direction='fwd', init_state=None):
        """
        x: [B, L, D]; init_state: [B, D] or None
        returns: y:[B,L,D], last_state:[B,D]
        """
        B, L, D = x.shape
        x = self.out_norm(x)
        gate = torch.sigmoid(self.gate(x))
        beta, gamma, delta = self.proj_in(x).chunk(3, dim=-1)
        beta = torch.sigmoid(beta)

        s = x.new_zeros(B, D) if init_state is None else init_state

        if direction == 'rev':
            x = torch.flip(x, [1])
            gate, beta, gamma, delta = map(lambda t: torch.flip(t, [1]),
                                           (gate, beta, gamma, delta))

        a = alpha_eff.squeeze(1)  # [1, D]
        ys = []
        for t in range(L):
            s = a.squeeze(0) * s + beta[:, t] * x[:, t]
            y_t = torch.tanh(gamma[:, t]) * s + delta[:, t]
            y_t = gate[:, t] * y_t + (1 - gate[:, t]) * x[:, t]
            ys.append(y_t)

        y = torch.stack(ys, 1)
        if direction == 'rev':
            y = torch.flip(y, [1])
        return y, s

    def forward(self, x, init_state_fwd=None, init_state_rev=None):
        """
        x: [B, L, D]
        returns: y:[B,L,D], last_state_fwd:[B,D], last_state_rev:[B,D or None]
        """
        alpha_eff = self._alpha_eff()

        y_f, s_f = self._scan_1d(x, alpha_eff, 'fwd', init_state_fwd)
        if not self.bidi:
            return self.drop(y_f), s_f, None

        y_b, s_b = self._scan_1d(x, alpha_eff, 'rev', init_state_rev)

        if self.fuse_mode == 'avg':
            y = 0.5 * (y_f + y_b)
        else:
            g = torch.sigmoid(self.fuse_gate(x))
            y = g * y_f + (1 - g) * y_b

        return self.drop(y), s_f, s_b


class AxialMambaBlock(nn.Module):
    def __init__(self, c, input_resolution,
                 appearance_guidance_dim=0,
                 chunk_len=12, ds=1, nlayers=1,
                 prior_heads=4,
                 prior_gamma_range=(0.6, 0.98),
                 prior_bidi=True,
                 cross_row_gain=1.0,
                 learnable_cross_gain=False):
        super().__init__()
        self.c = c
        self.H, self.W = input_resolution
        self.chunk_len = chunk_len
        self.ds = ds

        # cross-row gain (kept for compatibility; still used)
        self.cross_row_gain = nn.Parameter(torch.tensor(float(cross_row_gain))) \
            if learnable_cross_gain else float(cross_row_gain)

        # optional appearance-guided FiLM
        self.film = FiLM2D(appearance_guidance_dim, c) if appearance_guidance_dim > 0 else None

        # depthwise + pointwise conv before Mamba
        self.pre_mix = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c),
            nn.Conv2d(c, c, 1),
            nn.GELU()
        )

        # 1D Mamba along each row (batched over B*H)
        self.seq_row = Mamba1DWithDecayPrior(
            d_model=c, num_heads=prior_heads,
            gamma_range=prior_gamma_range, bidi=prior_bidi
        )

        self.post_proj = nn.Conv2d(c, c, 1)
        self.norm_tok = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, 3 * c), nn.GELU(), nn.Linear(3 * c, c))

        self.down = nn.AvgPool2d(ds, ds) if ds > 1 else nn.Identity()
        self.up = lambda x, s: F.interpolate(x, size=s, mode='bilinear', align_corners=True)

    @torch.no_grad()
    def _make_chunk_idx(self, L: int):
        """
        Compute chunk boundaries along width dimension.
        """
        idx = list(range(0, L, self.chunk_len)) if self.chunk_len > 0 else [0]
        if idx[-1] != L:
            idx.append(L)
        return idx

    def _scan_row_chunked_with_crossrow(self, x2d: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x2d.shape
        cuts = self._make_chunk_idx(W)
        out = x2d.new_empty(B, H, W, C)

        # state caches for in-row cross-chunk passing: [B,H,C]
        state_fwd = x2d.new_zeros(B, H, C)
        state_rev = x2d.new_zeros(B, H, C)

        # cross-row gain as scalar tensor
        gain = self.cross_row_gain if isinstance(self.cross_row_gain, torch.Tensor) \
            else x2d.new_tensor(self.cross_row_gain)

        # precompute odd row indices (if any)
        if H > 1:
            odd_rows = torch.arange(1, H, 2, device=x2d.device)
        else:
            odd_rows = None

        for s, e in zip(cuts[:-1], cuts[1:]):
            L = e - s
            # extract this chunk for all rows: [B,C,H,L] -> [B,H,L,C]
            seq_bhlc = x2d[:, :, :, s:e].permute(0, 2, 3, 1).contiguous()

            # build a snake-like version inside this chunk:
            #   even rows: as-is
            #   odd rows:  reversed along L dimension
            seq_snake = seq_bhlc.clone()
            if odd_rows is not None and odd_rows.numel() > 0:
                seq_snake[:, odd_rows] = torch.flip(seq_bhlc[:, odd_rows], dims=[2])  # flip over length dim

            # flatten over rows: [B,H,L,C] -> [B*H,L,C]
            seq = seq_snake.view(B * H, L, C)

            # merge row-wise initial states: [B,H,C] -> [B*H,C]
            init_f = state_fwd.view(B * H, C)
            init_r = state_rev.view(B * H, C)

            # run Mamba over this chunk for all rows in parallel
            y_flat, s_f_flat, s_r_flat = self.seq_row(
                seq, init_state_fwd=init_f, init_state_rev=init_r
            )

            # reshape back: [B*H,L,C] -> [B,H,L,C]
            y_bhlc = y_flat.view(B, H, L, C)

            # invert the snake flip for odd rows before writing outputs
            y_snake = y_bhlc.clone()
            if odd_rows is not None and odd_rows.numel() > 0:
                y_snake[:, odd_rows] = torch.flip(y_bhlc[:, odd_rows], dims=[2])

            # write to output buffer: [B,H,L,C] -> [B,H,W,C]
            out[:, :, s:e, :] = y_snake

            # recover per-row final states: [B*H,C] -> [B,H,C]
            s_f = s_f_flat.view(B, H, C)
            s_r = s_r_flat.view(B, H, C) if s_r_flat is not None else None

            # ---- cross-row passing (row i -> row i+1) ----
            s_f_down = torch.zeros_like(s_f)
            s_f_down[:, 1:, :] = s_f[:, :-1, :]  # row i's state goes into row i+1

            state_fwd = s_f + gain * s_f_down  # out-of-place update

            if s_r is not None:
                s_r_down = torch.zeros_like(s_r)
                s_r_down[:, 1:, :] = s_r[:, :-1, :]
                state_rev = s_r + gain * s_r_down
            else:
                state_rev = state_rev  # keep previous reverse state if not updated

        # [B,H,W,C] -> [B,C,H,W]
        return out.permute(0, 3, 1, 2).contiguous()

    def forward(self, x, app_guidance=None):
        """
        x: [B, C, T, H, W] -> output: [B, C, T, H, W]
        Inter-row interactions are modeled by cross-row state passing;
        within each row, we apply chunk-wise snake-like Mamba scanning.
        """
        B, C, T, H, W = x.shape
        xs = rearrange(x, 'B C T H W -> (B T) C H W')

        # optional FiLM with appearance guidance
        if self.film is not None and app_guidance is not None:
            # app_guidance: [B, Ca, H, W]
            app = app_guidance.unsqueeze(1).expand(B, T, app_guidance.size(1), H, W) \
                                          .reshape(B * T, app_guidance.size(1), H, W)
            xs = self.film(xs, app) + xs

        # optional low-resolution processing
        Hs, Ws = (H // self.ds, W // self.ds) if self.ds > 1 else (H, W)
        x_low = self.down(xs)
        z = self.pre_mix(x_low)

        # row-wise chunked scan with cross-row state passing + snake-like chunk pattern
        y2d = self._scan_row_chunked_with_crossrow(z)

        y2d = self.post_proj(y2d)
        y = x_low + y2d

        # token-wise FFN at low resolution
        y_ln = self.norm_tok(rearrange(y, '(B T) C H W -> (B T) (H W) C', B=B, T=T))
        y_ffn = self.ffn(y_ln)
        y = rearrange(y_ln + y_ffn, '(B T) (H W) C -> (B T) C H W', B=B, T=T, H=Hs, W=Ws)

        # upsample back (if ds > 1) and add residual
        y_up = y if self.ds == 1 else self.up(y, (H, W))
        out = xs + y_up
        return rearrange(out, '(B T) C H W -> B C T H W', B=B, T=T)




class PseudoMamba1D(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.d = d_model
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.SiLU()
        )
        self.proj_in  = nn.Linear(d_model, 3*d_model, bias=True)   # beta, gamma, delta
        self.alpha_logit = nn.Parameter(torch.zeros(d_model))      # alpha = sigmoid(alpha_logit) in (0,1)
        self.out_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, L, D]
        B, L, D = x.shape
        assert D == self.d
        x = self.out_norm(x)
        gate = torch.sigmoid(self.gate(x))# [B,L,D]
        beta, gamma, delta = self.proj_in(x).chunk(3, dim=-1)  # [B,L,D] ×3
        alpha = torch.sigmoid(self.alpha_logit).view(1,1,D)    # [1,1,D], (0,1)

        s = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            bt = torch.sigmoid(beta[:, t, :])
            gt = gate[:, t, :]
            s = alpha.squeeze(1) * s + bt * x[:, t, :]          # [B,D]
            yt = torch.tanh(gamma[:, t, :]) * s + delta[:, t, :]# [B,D]
            yt = gt * yt + (1 - gt) * x[:, t, :]                # selective 混合
            ys.append(yt)

        y = torch.stack(ys, dim=1)       # [B,L,D]
        return self.drop(y)


class _Mamba1DAdapterLite(nn.Module):
    def __init__(self, d_model, nlayers=1, nheads=4, dropout=0.0, backend='pseudo'):
        super().__init__()
        self.backend = backend
        blocks = []
        if backend == 'mamba':
            try:
                from mamba_ssm import Mamba
                for _ in range(nlayers):
                    blocks.append(Mamba(d_model=d_model))
            except Exception:
                print("[WARN] mamba-ssm cannot be imported")
                backend = 'pseudo'
        if backend == 'pseudo':
            for _ in range(nlayers):
                blocks.append(PseudoMamba1D(d_model, dropout=dropout))
        elif backend == 'transformer':
            enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nheads, batch_first=True, dropout=dropout)
            for _ in range(nlayers):
                blocks.append(nn.TransformerEncoder(enc, num_layers=1))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, seq):  # [Bseq, L, D]
        x = self.norm(seq)
        for b in self.blocks:
            x = b(x)
        return x


class ClassMambaLayer(nn.Module):
    def __init__(self, hidden_dim=64, guidance_dim=0,
                 pooling_size=(8, 8), pad_len=0, nlayers=1):
        super().__init__()
        self.pool = nn.AvgPool2d(pooling_size) if pooling_size is not None else nn.Identity()
        self.pad_len = pad_len
        self.guid_proj = nn.Linear(guidance_dim, hidden_dim * 2) if guidance_dim > 0 else None
        self.seq_block = _Mamba1DAdapterLite(d_model=hidden_dim, nlayers=nlayers)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, 3 * hidden_dim), nn.GELU(),
                                 nn.Linear(3 * hidden_dim, hidden_dim))

    def pool_features(self, x):
        B = x.size(0)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        x = self.pool(x)
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        return x

    def forward(self, x, guidance):  # x:[B,C,T,H,W], guidance:[B,T,Cg] or None
        B, C, T, H, W = x.size()
        x_pool = self.pool_features(x)  # [B,C,T,H',W']
        *_, Hp, Wp = x_pool.size()

        orig_len = T
        if self.pad_len and orig_len < self.pad_len:
            pad_T = self.pad_len - orig_len
            pad_tok = x_pool.new_zeros(B, C, pad_T, Hp, Wp)
            x_pool = torch.cat([x_pool, pad_tok], dim=2)
            Tp = self.pad_len
        else:
            Tp = T

        # [B,C,Tp,H',W'] -> [(B*H'*W'), Tp, C]
        x_seq = rearrange(x_pool, 'B C T H W -> (B H W) T C')

        if guidance is not None and self.guid_proj is not None:
            if self.pad_len and orig_len < self.pad_len:
                pad_g = guidance.new_zeros(B, self.pad_len - orig_len, guidance.shape[-1])
                guidance = torch.cat([guidance, pad_g], dim=1)  # [B,Tp,Cg]
            gam_beta = self.guid_proj(guidance)  # [B,Tp,2C]
            # expand 到 (B*H'*W',Tp,C)
            gam, bet = torch.chunk(gam_beta, 2, dim=-1)
            gam = gam.unsqueeze(1).unsqueeze(1).expand(B, Hp, Wp, Tp, C).reshape(B * Hp * Wp, Tp, C)
            bet = bet.unsqueeze(1).unsqueeze(1).expand(B, Hp, Wp, Tp, C).reshape(B * Hp * Wp, Tp, C)
            x_seq = x_seq * torch.sigmoid(gam) + bet

        z = self.seq_block(self.norm1(x_seq))
        z = z + x_seq
        z = z + self.ffn(self.norm2(z))

        x_pool2 = rearrange(z, '(B H W) T C -> (B T) C H W', B=B, H=Hp, W=Wp)
        x_pool2 = F.interpolate(x_pool2, size=(H, W), mode='bilinear', align_corners=True)
        x_pool2 = rearrange(x_pool2, '(B T) C H W -> B C T H W', B=B)
        if self.pad_len and orig_len < self.pad_len:
            x_pool2 = x_pool2[:, :, :orig_len]
        return x + x_pool2


class UpLite(nn.Module):
    """ConvTranspose2d + single concatenation with guidance"""
    def __init__(self, in_channels, out_channels, guidance_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels - guidance_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, out_channels // 16), out_channels),
            nn.ReLU(inplace=False),   # use inplace=False for stability
        )

    def forward(self, x, guidance=None):
        x = self.up(x)
        if guidance is not None:
            Btg, Cg, Hg, Wg = guidance.shape
            Bt = x.size(0) // Btg
            g = guidance.unsqueeze(1).expand(Btg, Bt, Cg, Hg, Wg).reshape(Btg * Bt, Cg, Hg, Wg)
            x = torch.cat([x, g], dim=1)
        return self.conv(x)


class S2_Corr(nn.Module):
    def __init__(self,
                 text_guidance_dim=512,
                 text_guidance_proj_dim=128,
                 appearance_guidance_dim=512,
                 appearance_guidance_proj_dim=128,
                 decoder_dims=(64, 32),
                 decoder_guidance_dims=(256, 128),
                 decoder_guidance_proj_dims=(32, 16),
                 num_layers=4,
                 hidden_dim=128,
                 pooling_size=(8, 8),
                 feature_resolution=(24, 24),
                 chunk_size=12,
                 pad_len=0,
                 cross_row_gain=1.0,
                 learnable_cross_gain=False,
                 nheads=1,
                 gamma_range=(0.5, 0.6),
                 **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pad_len = pad_len

        self.layers = nn.ModuleList([
            nn.Sequential(
                AxialMambaBlock(
                    c=hidden_dim,
                    input_resolution=feature_resolution,
                    appearance_guidance_dim=appearance_guidance_proj_dim,
                    chunk_len=chunk_size, nlayers=self.num_layers,
                    prior_heads=nheads, prior_gamma_range=gamma_range, prior_bidi=False,
                    cross_row_gain=cross_row_gain, learnable_cross_gain=learnable_cross_gain
                ),
                ClassMambaLayer(
                    hidden_dim=hidden_dim,
                    guidance_dim=text_guidance_proj_dim,
                    pooling_size=pooling_size,
                    pad_len=pad_len, nlayers=self.num_layers,
                )
            ) for _ in range(num_layers)
        ])

        self.conv1 = nn.Conv2d(1, hidden_dim, kernel_size=7, stride=1, padding=3)

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, 3, 1, 1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None

        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, 3, 1, 1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        # decoder
        self.decoder1 = UpLite(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = UpLite(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def correlation(img_feats, text_feats):
        """
        img_feats: [B, C, H, W]
        text_feats: [B, T, P, C]
        returns corr: [B,1,T,H,W] (P fused into 1 via mean)
        """
        img_feats = F.normalize(img_feats, dim=1)
        text_feats = F.normalize(text_feats, dim=-1)
        text_fused = text_feats.mean(dim=2)  # [B, T, C]
        corr = torch.einsum('bchw, btc -> bthw', img_feats, text_fused).unsqueeze(1)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        x2d = rearrange(x, 'B 1 T H W -> (B T) 1 H W')
        x2d = self.conv1(x2d)
        x3d = rearrange(x2d, '(B T) C H W -> B C T H W', B=B)
        return x3d

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        x2d = rearrange(x, 'B C T H W -> (B T) C H W')
        x2d = self.decoder1(x2d, guidance[0] if guidance and guidance[0] is not None else None)
        x2d = self.decoder2(x2d, guidance[1] if guidance and guidance[1] is not None else None)
        x2d = self.head(x2d)
        out = rearrange(x2d, '(B T) () H W -> B T H W', B=B)
        return out

    def forward(self, img_feats, text_feats, appearance_guidance):
        # 1) correlation map (P -> 1)
        corr = self.correlation(img_feats, text_feats)    # [B,1,T,H,W]
        corr_embed = self.corr_embed(corr)                # [B,C,T,H,W]

        # 2) guidance projection
        projected_guidance, projected_text_guidance = None, None
        projected_decoder_guidance = [None, None]

        if self.guidance_projection is not None and appearance_guidance[0] is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])

        if self.decoder_guidance_projection is not None and appearance_guidance[1] is not None:
            projected_decoder_guidance = [
                proj(g) if g is not None else None
                for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])
            ]

        if self.text_guidance_projection is not None:
            t = text_feats.mean(dim=2)                       # [B,T,C]
            t = t / (t.norm(dim=-1, keepdim=True) + 1e-6)
            projected_text_guidance = self.text_guidance_projection(t)  # [B,T,C']

        x = corr_embed
        for layer in self.layers:
            spatial, class_layer = layer[0], layer[1]
            x = spatial(x, projected_guidance)              # row-only + cross-row state passing
            x = class_layer(x, projected_text_guidance)

        logit = self.conv_decoder(x, projected_decoder_guidance)
        return logit
