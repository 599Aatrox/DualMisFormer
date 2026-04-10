import logging
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.RevIN import RevIN
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer

# 创建 logs 目录（如果不存在）
os.makedirs("logs", exist_ok=True)

# 配置日志
logging.basicConfig(
    filename='logs/experiment.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()
warnings.filterwarnings('ignore')


class Model(nn.Module):
    """
    DualMixFormer + Lite MSMR Residual Branch
    MSMR = Multi-Scale + Multi-Resolution
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # ===== basic =====
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in

        # 兼容 cycle / cycle_len 两种写法
        self.cycle_len = getattr(configs, 'cycle_len', getattr(configs, 'cycle', 24))

        self.use_ME = getattr(configs, 'use_ME', 1)   # 多嵌入
        self.use_L = getattr(configs, 'use_L', 1)     # 线性低频分支
        self.use_T = getattr(configs, 'use_T', 1)     # Transformer分支
        self.use_R = getattr(configs, 'use_R', 1)     # residual升级开关

        # ===== residual branch hyper-params =====
        self.top_k = getattr(configs, 'top_k', 2)
        self.num_scales = getattr(configs, 'ms_scales', 3)  # 不改 parser 也能跑
        self.min_scale_len = 8

        self.ma_kernel = getattr(configs, 'moving_avg', 25)
        if self.ma_kernel % 2 == 0:
            self.ma_kernel += 1

        # ===== embedding + encoder =====
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # ===== MEVA =====
        if self.use_ME:
            self.channel_embedding = nn.Parameter(torch.zeros(self.enc_in, self.d_model))
            self.phase_embedding = nn.Embedding(self.cycle_len, self.d_model)

            nn.init.xavier_normal_(self.channel_embedding)
            nn.init.xavier_normal_(self.phase_embedding.weight)

        # ===== projector =====
        self.projector = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model * 2),
            nn.GELU(),
            nn.Dropout(configs.output_proj_dropout),
            nn.Linear(configs.d_model * 2, configs.d_model * 4),
            nn.GELU(),
            nn.Dropout(configs.output_proj_dropout),
            nn.Linear(configs.d_model * 4, configs.pred_len),
        )

        # ===== low-frequency / smooth branch =====
        if self.use_L:
            self.Linear = nn.Linear(self.seq_len, self.seq_len)
            self.GeLU = nn.GELU()
            self.Hidden1 = nn.Linear(self.seq_len, self.pred_len)
            self.w_dec = nn.Parameter(
                torch.FloatTensor([configs.w_lin] * configs.enc_in),
                requires_grad=True
            )

        # ===== Lite MSMR residual branch =====
        # 输入按 [B, C, P, F] 做 depthwise Conv2d，参数量很小
        if self.use_R:
            self.res_season_mixer = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.enc_in,
                    out_channels=self.enc_in,
                    kernel_size=(1, 3),
                    padding=(0, 1),
                    groups=self.enc_in
                ),
                nn.GELU(),
                nn.Conv2d(
                    in_channels=self.enc_in,
                    out_channels=self.enc_in,
                    kernel_size=1,
                    groups=self.enc_in
                )
            )

            self.res_trend_mixer = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.enc_in,
                    out_channels=self.enc_in,
                    kernel_size=(3, 1),
                    padding=(1, 0),
                    groups=self.enc_in
                ),
                nn.GELU(),
                nn.Conv2d(
                    in_channels=self.enc_in,
                    out_channels=self.enc_in,
                    kernel_size=1,
                    groups=self.enc_in
                )
            )

            # 多尺度融合权重
            self.scale_logits = nn.Parameter(torch.zeros(self.num_scales))
            # residual block强度门控
            self.res_alpha = nn.Parameter(torch.tensor(0.5))

        self.revin_layer = RevIN(configs.enc_in)
        self.log_prem = True

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, phase,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # ---------------------------
        # 1) RevIN normalize
        # ---------------------------
        x_enc = self.revin_layer(x_enc, 'norm')   # [B, L, N]
        B, L, N = x_enc.shape

        # ---------------------------
        # 2) explicit low/high split
        # ---------------------------
        if self.use_R:
            c_low_hist = self.moving_avg(x_enc)           # [B, L, N]
            r_hist = x_enc - c_low_hist                   # [B, L, N]
            r_hist = self.ms_mr_residual_block(r_hist)    # [B, L, N]
            low_branch_input = c_low_hist
            trans_branch_input = r_hist
        else:
            c_low_hist = None
            low_branch_input = x_enc
            trans_branch_input = x_enc

        # ---------------------------
        # 3) Transformer residual branch
        # ---------------------------
        enc_out = self.enc_embedding(trans_branch_input, x_mark_enc)  # 你当前写法默认是 [B, N, d_model]

        if self.log_prem:
            logger.info(f"Input shape - B: {B}, L: {L}, N: {N}")
            logger.info(f"Transformer branch input shape: {trans_branch_input.shape}")
            logger.info(f"Encoded output shape: {enc_out.shape}")

        if self.use_ME:
            phase = phase.long().view(B) % self.cycle_len
            channel_emb = self.channel_embedding.unsqueeze(0).expand(B, -1, -1)      # [B, N, d_model]
            phase_emb = self.phase_embedding(phase).unsqueeze(1).expand(-1, N, -1)   # [B, N, d_model]

            enc_out = enc_out[:, :N, :] + channel_emb + phase_emb

            if self.log_prem:
                logger.info(f"Channel embedding shape: {channel_emb.shape}")
                logger.info(f"Phase embedding shape: {phase_emb.shape}")
                logger.info(f"Final encoded output shape: {enc_out.shape}")

        enc_origin = enc_out
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        if self.log_prem:
            logger.info(f"Encoder output shape: {enc_out.shape}")

        trans_out = self.projector(enc_out + enc_origin).permute(0, 2, 1)[:, :, :N]   # [B, pred_len, N]
        trans_out = trans_out[:, -self.pred_len:, :]

        if self.log_prem:
            logger.info(f"Transformer output shape: {trans_out.shape}")

        # ---------------------------
        # 4) Low-frequency branch
        # ---------------------------
        low_out = None
        if self.use_L:
            low_out = self.low_branch_forward(low_branch_input)   # [B, pred_len, N]

            if self.log_prem:
                logger.info(f"Low branch output shape: {low_out.shape}")
        else:
            # 如果 use_R=1 但不用线性分支，至少给一个最朴素的低频基底
            if self.use_R and c_low_hist is not None:
                low_out = c_low_hist[:, -1:, :].repeat(1, self.pred_len, 1)

        # ---------------------------
        # 5) Fusion
        # ---------------------------
        if self.use_L:
            if self.use_T:
                raw_out = trans_out + self.w_dec.view(1, 1, -1) * low_out
            else:
                raw_out = low_out
        else:
            if self.use_T:
                if low_out is not None:
                    raw_out = trans_out + low_out
                else:
                    raw_out = trans_out
            else:
                raw_out = low_out if low_out is not None else trans_out

        dec_out = self.revin_layer(raw_out, 'denorm')

        if self.log_prem:
            logger.info(f"Final decoder output shape: {dec_out.shape}")

        self.log_prem = False

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]

    # =========================================================
    # Low-frequency branch
    # =========================================================
    def low_branch_forward(self, x):
        """
        x: [B, L, N]
        return: [B, pred_len, N]
        """
        x = x.permute(0, 2, 1)   # [B, N, L]
        x = self.Linear(x)
        x = self.GeLU(x)
        x = self.Hidden1(x)
        x = x.permute(0, 2, 1)   # [B, pred_len, N]
        return x

    # =========================================================
    # Moving average
    # =========================================================
    def moving_avg(self, x):
        """
        x: [B, L, N]
        return: [B, L, N]
        """
        B, L, N = x.shape
        x = x.permute(0, 2, 1).reshape(B * N, 1, L)   # [B*N, 1, L]

        pad = (self.ma_kernel - 1) // 2
        if pad > 0:
            front = x[:, :, 0:1].repeat(1, 1, pad)
            end = x[:, :, -1:].repeat(1, 1, pad)
            x = torch.cat([front, x, end], dim=-1)

        avg = F.avg_pool1d(x, kernel_size=self.ma_kernel, stride=1)   # [B*N, 1, L]
        avg = avg.reshape(B, N, L).permute(0, 2, 1)                   # [B, L, N]
        return avg

    # =========================================================
    # Multi-scale + Multi-resolution residual block
    # =========================================================
    def ms_mr_residual_block(self, residual):
        """
        residual: [B, L, N]
        return:   [B, L, N]
        """
        B, L, N = residual.shape
        scales = self.build_scales(residual)   # list of [B, Lm, N]

        # 用最粗尺度检测 top-k periods
        periods, res_weights = self.detect_topk_periods(scales[-1])

        if self.log_prem:
            logger.info(f"Detected periods: {periods}")

        scale_feats = []

        for s in scales:
            # s: [B, Lm, N] -> [B, N, Lm]
            s_ncl = s.permute(0, 2, 1).contiguous()
            mr_feats = []

            for p in periods:
                img, orig_len = self.to_time_image(s_ncl, p)   # [B, N, P, F]
                season = self.res_season_mixer(img)
                trend = self.res_trend_mixer(img)

                feat = self.from_time_image(season + trend, orig_len)   # [B, N, Lm]
                mr_feats.append(feat)

            if len(mr_feats) == 1:
                fused_mr = mr_feats[0]
            else:
                fused_mr = torch.zeros_like(mr_feats[0])
                for w, feat in zip(res_weights, mr_feats):
                    fused_mr = fused_mr + w * feat

            # 每个尺度保留 residual shortcut
            fused_mr = fused_mr + s_ncl
            scale_feats.append(fused_mr.permute(0, 2, 1).contiguous())   # [B, Lm, N]

        # 多尺度融合
        scale_weights = torch.softmax(self.scale_logits[:len(scale_feats)], dim=0)

        out = torch.zeros_like(residual)
        for w, feat in zip(scale_weights, scale_feats):
            if feat.size(1) != L:
                feat = self.upsample_sequence(feat, L)
            out = out + w * feat

        gate = torch.sigmoid(self.res_alpha)
        return residual + gate * out

    def build_scales(self, x):
        """
        x: [B, L, N]
        return: list of [B, Lm, N]
        """
        scales = [x]
        cur = x
        for _ in range(1, self.num_scales):
            if cur.size(1) < self.min_scale_len:
                break
            cur = self.downsample_sequence(cur)
            scales.append(cur)
        return scales

    def downsample_sequence(self, x):
        """
        x: [B, L, N]
        return: [B, ceil(L/2), N]
        """
        x = x.permute(0, 2, 1)                    # [B, N, L]
        x = F.avg_pool1d(x, kernel_size=2, stride=2, ceil_mode=True)
        x = x.permute(0, 2, 1)                    # [B, L/2, N]
        return x

    def upsample_sequence(self, x, target_len):
        """
        x: [B, Lm, N]
        return: [B, target_len, N]
        """
        x = x.permute(0, 2, 1)   # [B, N, Lm]
        x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        x = x.permute(0, 2, 1)
        return x

    def detect_topk_periods(self, x):
        """
        x: [B, L, N]
        从最粗尺度检测 top-k 周期
        """
        x = x.permute(0, 2, 1).contiguous()   # [B, N, L]
        L = x.size(-1)

        if L < 4:
            periods = [max(2, L)]
            weights = x.new_tensor([1.0])
            return periods, weights

        spec = torch.fft.rfft(x, dim=-1)              # [B, N, F]
        amp = spec.abs().mean(dim=(0, 1))             # [F]
        amp[0] = 0.0                                  # 去掉直流分量

        valid_k = min(self.top_k, max(1, amp.numel() - 1))
        top_vals, top_idx = torch.topk(amp[1:], k=valid_k)
        top_idx = top_idx + 1

        periods = []
        raw_weights = []

        for val, idx in zip(top_vals, top_idx):
            # 周期 = 序列长度 / 频率索引
            period = int(round(L / float(idx.item())))
            period = max(2, min(L, period))

            if period not in periods:
                periods.append(period)
                raw_weights.append(val)

        if len(periods) == 0:
            periods = [max(2, min(L, self.ma_kernel))]
            weights = x.new_tensor([1.0])
        else:
            weights = torch.softmax(torch.stack(raw_weights), dim=0)

        return periods, weights

    def to_time_image(self, x, period):
        """
        x: [B, N, L]
        return img: [B, N, P, F], orig_len
        """
        B, N, L = x.shape
        pad_len = (period - L % period) % period

        if pad_len > 0:
            x = F.pad(x, (0, pad_len), mode='replicate')

        freq = x.size(-1) // period
        img = x.view(B, N, period, freq)
        return img, L

    def from_time_image(self, img, orig_len):
        """
        img: [B, N, P, F]
        return: [B, N, L]
        """
        B, N, P, Freq = img.shape
        x = img.reshape(B, N, P * Freq)
        x = x[:, :, :orig_len]
        return x