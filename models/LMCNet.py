import math
import logging
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN

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


def _parse_scales(scales):
    """
    支持:
    - list: [1,2,4]
    - str : "1,2,4"
    """
    if scales is None:
        return [1, 2, 4]
    if isinstance(scales, str):
        scales = scales.replace(' ', '')
        return [int(x) for x in scales.split(',') if x != '']
    return list(scales)


class MultiKernelLowFreqExtractor(nn.Module):
    """
    多核平滑 + 动态选择

    输入:
        x: [B, L, C]
    输出:
        x_low: [B, L, C]
        gates: [B, C, K]
    """
    def __init__(self, enc_in, kernels=(5, 13, 25), gate_hidden=16):
        super().__init__()
        self.enc_in = enc_in
        self.kernels = list(kernels)
        self.num_kernels = len(self.kernels)

        self.dw_convs = nn.ModuleList()
        for k in self.kernels:
            conv = nn.Conv1d(
                in_channels=enc_in,
                out_channels=enc_in,
                kernel_size=k,
                padding=k // 2,
                groups=enc_in,
                bias=False
            )
            # 初始化成平均滤波
            nn.init.constant_(conv.weight, 1.0 / k)
            self.dw_convs.append(conv)

        # 对每个变量的时间均值 -> 生成该变量对 K 个平滑核的权重
        # 输入 [B, C, 1]，输出 [B, C, K]
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.num_kernels)
        )

    def forward(self, x):
        # x: [B, L, C]
        x_t = x.transpose(1, 2)  # [B, C, L]

        smooth_list = []
        for conv in self.dw_convs:
            smooth = conv(x_t).transpose(1, 2)  # [B, L, C]
            smooth_list.append(smooth)

        # [B, L, C, K]
        smooth_stack = torch.stack(smooth_list, dim=-1)

        # [B, C] -> [B, C, 1]
        mean_time = x.mean(dim=1).unsqueeze(-1)
        gates = torch.softmax(self.gate_mlp(mean_time), dim=-1)  # [B, C, K]

        # 对 K 个核做加权求和
        x_low = (smooth_stack * gates.unsqueeze(1)).sum(dim=-1)  # [B, L, C]
        return x_low, gates


class LowFreqHead(nn.Module):
    """
    低频分支:
    共享 temporal linear (L -> P) + per-channel affine

    输入:
        x_low: [B, L, C]
    输出:
        y_low: [B, P, C]
    """
    def __init__(self, seq_len, pred_len, enc_in):
        super().__init__()
        self.shared_temporal = nn.Linear(seq_len, pred_len)
        self.channel_scale = nn.Parameter(torch.ones(1, 1, enc_in))
        self.channel_bias = nn.Parameter(torch.zeros(1, 1, enc_in))

    def forward(self, x_low):
        x = x_low.transpose(1, 2)                    # [B, C, L]
        y = self.shared_temporal(x).transpose(1, 2) # [B, P, C]
        y = y * self.channel_scale + self.channel_bias
        return y


class TemporalBlock(nn.Module):
    """
    每个尺度上的轻量 temporal encoder block

    输入:
        x: [B, Ls, C]
    输出:
        out: [B, Ls, C]
    """
    def __init__(self, enc_in, kernel_size=3, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        hidden = max(enc_in, int(enc_in * mlp_ratio))

        self.dw_conv = nn.Conv1d(
            in_channels=enc_in,
            out_channels=enc_in,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=enc_in,
            bias=True
        )
        self.norm1 = nn.LayerNorm(enc_in)

        self.ffn = nn.Sequential(
            nn.Linear(enc_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, enc_in),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(enc_in)

    def forward(self, x):
        # x: [B, Ls, C]
        residual = x
        y = self.dw_conv(x.transpose(1, 2)).transpose(1, 2)  # [B, Ls, C]
        y = self.norm1(y + residual)

        residual = y
        y = self.ffn(y)
        y = self.norm2(y + residual)
        return y


class ScaleTemporalEncoder(nn.Module):
    """
    每尺度:
    residual -> TempBlock -> variable token

    输入:
        r_s: [B, Ls, C]
    输出:
        h_s: [B, C, d_model]
    """
    def __init__(self, enc_in, scale_len, d_model, kernel_size=3, dropout=0.1):
        super().__init__()
        self.temp_block = TemporalBlock(
            enc_in=enc_in,
            kernel_size=kernel_size,
            mlp_ratio=2.0,
            dropout=dropout
        )
        self.token_proj = nn.Linear(scale_len, d_model)

    def forward(self, x):
        u = self.temp_block(x)               # [B, Ls, C]
        h = self.token_proj(u.transpose(1, 2))  # [B, C, d_model]
        return h


class DynamicVarCoupling(nn.Module):
    """
    动态变量耦合模块

    输入:
        h:         [B, C, d_model]
        low_ctx:   [B, C, d_model]
        scale_emb: [d_model]

    输出:
        z:         [B, C, d_model]
        attn:      [B, C, C]
    """
    def __init__(self, d_model, num_vars, rel_rank=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.rel_rank = rel_rank

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # 静态关系偏置: 低秩
        self.static_u = nn.Parameter(torch.randn(num_vars, rel_rank) * 0.02)
        self.static_v = nn.Parameter(torch.randn(num_vars, rel_rank) * 0.02)

        # 动态关系偏置
        self.dyn_a = nn.Linear(d_model, rel_rank)
        self.dyn_b = nn.Linear(d_model, rel_rank)

        # 关系门控
        self.gate_a = nn.Linear(d_model, rel_rank)
        self.gate_b = nn.Linear(d_model, rel_rank)

        self.gamma = nn.Parameter(torch.tensor(1.0))  # low_ctx 条件系数
        self.lam = nn.Parameter(torch.tensor(0.5))    # static bias 系数
        self.mu = nn.Parameter(torch.tensor(0.5))     # dynamic bias 系数

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, h, low_ctx, scale_emb):
        # 条件化表示
        h_cond = h + scale_emb.view(1, 1, -1) + self.gamma * low_ctx  # [B, C, d_model]

        q = self.q_proj(h_cond)
        k = self.k_proj(h_cond)
        v = self.v_proj(h_cond)

        # 基础 attention
        a_base = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_model)  # [B, C, C]

        # 静态偏置
        b_static = torch.matmul(self.static_u, self.static_v.transpose(0, 1))    # [C, C]
        b_static = b_static.unsqueeze(0)                                           # [1, C, C]

        # 动态偏置
        a_dyn = self.dyn_a(h_cond)  # [B, C, r]
        b_dyn = self.dyn_b(h_cond)  # [B, C, r]
        b_dynamic = torch.matmul(a_dyn, b_dyn.transpose(-1, -2)) / math.sqrt(self.rel_rank)  # [B, C, C]

        # 门控矩阵
        g1 = self.gate_a(h_cond)
        g2 = self.gate_b(h_cond)
        gate = torch.sigmoid(torch.matmul(g1, g2.transpose(-1, -2)) / math.sqrt(self.rel_rank))  # [B, C, C]

        score = a_base + self.lam * b_static + self.mu * b_dynamic
        attn = torch.softmax(score, dim=-1)
        attn = self.dropout(attn)

        z = torch.matmul(attn * gate, v)  # [B, C, d_model]
        z = self.out_proj(z)
        z = self.norm1(z + h)

        z2 = self.ffn(z)
        z = self.norm2(z + z2)
        return z, attn


class ScaleFusion(nn.Module):
    """
    动态尺度融合

    输入:
        preds: list[[B, P, C]]
        summaries: list[[B, d_model]]
        low_summary: [B, d_model]

    输出:
        fused_pred: [B, P, C]
        alpha: [B, S]
    """
    def __init__(self, d_model, num_scales, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or max(d_model, 64)

        self.score_mlp = nn.Sequential(
            nn.Linear((num_scales + 1) * d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_scales)
        )

    def forward(self, preds, summaries, low_summary):
        fusion_input = torch.cat(summaries + [low_summary], dim=-1)  # [B, (S+1)*d]
        alpha = torch.softmax(self.score_mlp(fusion_input), dim=-1)  # [B, S]

        fused = 0.0
        for i, pred in enumerate(preds):
            fused = fused + alpha[:, i].view(-1, 1, 1) * pred
        return fused, alpha


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.output_attention = getattr(configs, 'output_attention', False)

        # 保留原有名字，重新解释成 ablation 开关
        self.use_ME = getattr(configs, 'use_ME', 0)   # channel/phase embedding
        self.use_L = getattr(configs, 'use_L', 1)     # low branch on/off
        self.use_T = getattr(configs, 'use_T', 1)     # residual branch on/off

        # 兼容 cycle / cycle_len 两种写法
        self.cycle_len = getattr(configs, 'cycle', getattr(configs, 'cycle_len', 24))

        self.ms_scales = _parse_scales(getattr(configs, 'ms_scales', [1, 2, 4]))
        self.low_kernels = _parse_scales(getattr(configs, 'low_kernels', [5, 13, 25]))

        self.rel_rank = getattr(configs, 'rel_rank', 16)
        self.temp_kernel = getattr(configs, 'temp_kernel', 3)
        self.smooth_loss_weight = getattr(configs, 'smooth_loss_weight', 0.0)
        self.dropout = getattr(configs, 'dropout', 0.1)


        # RevIN
        self.revin_layer = RevIN(self.enc_in)

        # 1) 低频骨架提取器
        self.low_extractor = MultiKernelLowFreqExtractor(
            enc_in=self.enc_in,
            kernels=self.low_kernels,
            gate_hidden=max(8, self.enc_in)
        )

        # 2) 低频预测头
        self.low_head = LowFreqHead(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            enc_in=self.enc_in
        )

        # 3) low branch context -> residual branch condition
        self.low_context_proj = nn.Linear(self.seq_len, self.d_model)

        # 4) 可选 channel / phase embedding
        if self.use_ME:
            self.channel_embedding = nn.Parameter(torch.zeros(self.enc_in, self.d_model))
            self.phase_embedding = nn.Embedding(self.cycle_len, self.d_model)
            nn.init.xavier_normal_(self.phase_embedding.weight)
            nn.init.xavier_normal_(self.channel_embedding)

        # 5) residual 多尺度模块
        self.scale_lengths = [max(1, self.seq_len // s) for s in self.ms_scales]
        self.scale_embeddings = nn.Parameter(torch.randn(len(self.ms_scales), self.d_model) * 0.02)

        self.temporal_encoders = nn.ModuleList()
        self.var_couplers = nn.ModuleList()
        self.pred_heads = nn.ModuleList()

        for scale_len in self.scale_lengths:
            self.temporal_encoders.append(
                ScaleTemporalEncoder(
                    enc_in=self.enc_in,
                    scale_len=scale_len,
                    d_model=self.d_model,
                    kernel_size=self.temp_kernel,
                    dropout=self.dropout
                )
            )
            self.var_couplers.append(
                DynamicVarCoupling(
                    d_model=self.d_model,
                    num_vars=self.enc_in,
                    rel_rank=self.rel_rank,
                    dropout=self.dropout
                )
            )
            self.pred_heads.append(nn.Linear(self.d_model, self.pred_len))

        # 6) 多尺度融合
        self.scale_fusion = ScaleFusion(
            d_model=self.d_model,
            num_scales=len(self.ms_scales),
            hidden_dim=max(64, self.d_model)
        )

        self.latest_aux_loss = torch.tensor(0.0)
        self.latest_scale_weight = None
        self.log_prem = True

    def _downsample(self, x, scale):
        """
        x: [B, L, C]
        return: [B, floor(L/scale), C]
        """
        if scale == 1:
            return x
        x_t = x.transpose(1, 2)  # [B, C, L]
        x_t = F.avg_pool1d(x_t, kernel_size=scale, stride=scale)
        return x_t.transpose(1, 2)

    def _smooth_loss(self, x_low):
        """
        x_low: [B, L, C]
        """
        if x_low.size(1) < 3:
            return x_low.new_tensor(0.0)
        second_diff = x_low[:, 2:, :] - 2 * x_low[:, 1:-1, :] + x_low[:, :-2, :]
        return second_diff.abs().mean()

    def get_aux_loss(self):
        return self.latest_aux_loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, phase,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # --------------------------------------------------
        # 0) RevIN
        # --------------------------------------------------
        x_norm = self.revin_layer(x_enc, 'norm')   # [B, L, C]
        B, L, C = x_norm.shape

        if self.log_prem:
            logger.info(f"Input shape - B: {B}, L: {L}, C: {C}")

        # --------------------------------------------------
        # 1) low-frequency skeleton
        # --------------------------------------------------
        x_low, low_gates = self.low_extractor(x_norm)   # [B, L, C], [B, C, K]
        residual = x_norm - x_low                       # [B, L, C]

        # low branch prediction
        y_low = self.low_head(x_low)                    # [B, P, C]

        # low context for residual branch
        low_ctx = self.low_context_proj(x_low.transpose(1, 2))  # [B, C, d_model]

        # optional channel / phase embedding
        if self.use_ME:
            channel_emb = self.channel_embedding.unsqueeze(0).expand(B, -1, -1)  # [B, C, d_model]

            if phase is None:
                phase = torch.zeros(B, device=x_norm.device, dtype=torch.long)

            phase = phase.view(B).long().clamp_(0, self.cycle_len - 1)
            phase_emb = self.phase_embedding(phase).unsqueeze(1).expand(-1, C, -1)  # [B, C, d_model]

            low_ctx = low_ctx + channel_emb + phase_emb
        else:
            channel_emb, phase_emb = None, None

        # --------------------------------------------------
        # 2) residual multi-scale branch
        # --------------------------------------------------
        scale_preds = []
        scale_summaries = []
        attn_list = []

        for i, scale in enumerate(self.ms_scales):
            r_s = self._downsample(residual, scale)          # [B, Ls, C]
            h_s = self.temporal_encoders[i](r_s)             # [B, C, d_model]

            if self.use_ME:
                h_s = h_s + channel_emb + phase_emb

            z_s, attn_s = self.var_couplers[i](
                h=h_s,
                low_ctx=low_ctx,
                scale_emb=self.scale_embeddings[i]
            )                                                # [B, C, d_model], [B, C, C]

            y_res_s = self.pred_heads[i](z_s).transpose(1, 2)  # [B, P, C]

            scale_preds.append(y_res_s)
            scale_summaries.append(z_s.mean(dim=1))         # [B, d_model]
            attn_list.append(attn_s)

        y_res, alpha = self.scale_fusion(
            preds=scale_preds,
            summaries=scale_summaries,
            low_summary=low_ctx.mean(dim=1)
        )                                                   # [B, P, C], [B, S]

        # --------------------------------------------------
        # 3) ablation switches
        # use_L: low branch
        # use_T: residual branch
        # --------------------------------------------------
        if self.use_L and self.use_T:
            dec_out = y_low + y_res
        elif self.use_L and not self.use_T:
            dec_out = y_low
        elif (not self.use_L) and self.use_T:
            dec_out = y_res
        else:
            # 两个都关时，默认仍输出完整结果，避免训练直接坏掉
            dec_out = y_low + y_res

        # --------------------------------------------------
        # 4) RevIN denorm
        # --------------------------------------------------
        dec_out = self.revin_layer(dec_out, 'denorm')

        # 可选辅助损失
        self.latest_aux_loss = self.smooth_loss_weight * self._smooth_loss(x_low)
        self.latest_scale_weight = alpha.detach()

        if self.log_prem:
            logger.info(f"x_low shape: {x_low.shape}")
            logger.info(f"residual shape: {residual.shape}")
            logger.info(f"y_low shape: {y_low.shape}")
            logger.info(f"y_res shape: {y_res.shape}")
            logger.info(f"Output shape: {dec_out.shape}")
            self.log_prem = False

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attn_list
        return dec_out[:, -self.pred_len:, :]