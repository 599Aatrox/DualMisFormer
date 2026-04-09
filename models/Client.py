import logging
import os
import warnings

import torch
import torch.nn as nn

from layers.Embed import DataEmbedding_inverted
from layers.RevIN import RevIN
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer

# 创建 logs 目录（如果不存在）
os.makedirs("logs", exist_ok=True)

# 配置日志
logging.basicConfig(
    filename='logs/experiment.log',  # 日志文件路径
    filemode='a',  # 追加模式（'w' 会覆盖）
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # 只记录 INFO 及以上级别
)
logger = logging.getLogger()
warnings.filterwarnings('ignore')


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # print(configs.d_model)
        # todo: d_model 最好是256但是作者给直接和seq_len一样了
        # configs.d_model = configs.seq_len


        self.seq_len = configs.seq_len
        # self.use_norm = configs.use_norm
        self.d_model = configs.d_model
        self.cycle_len = configs.cycle
        self.enc_in = configs.enc_in
        self.use_ME = configs.use_ME  # 使用多嵌入
        self.use_L = configs.use_L  # 使用使用线性分支
        self.use_T = configs.use_T
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # 通道嵌入
        if self.use_ME:
            self.channel_embedding = nn.Parameter(torch.zeros(configs.enc_in, configs.d_model))
            self.phase_embedding = nn.Embedding(self.cycle_len, configs.d_model)
            self.joint_embedding = nn.Embedding(self.cycle_len, self.enc_in * self.d_model)
            nn.init.xavier_normal_(self.phase_embedding.weight)
            nn.init.xavier_normal_(self.joint_embedding.weight)

        self.projector = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model * 2),
            nn.GELU(),
            nn.Dropout(configs.output_proj_dropout),
            nn.Linear(configs.d_model * 2, configs.d_model * 4),
            nn.GELU(),
            nn.Dropout(configs.output_proj_dropout),
            nn.Linear(configs.d_model * 4, configs.pred_len),
        )
        if self.use_L:
            self.Linear = nn.Linear(self.seq_len, self.seq_len)
            self.GeLU = nn.GELU()
            self.Hidden1 = nn.Linear(self.seq_len, self.pred_len)
            self.w_dec = torch.nn.Parameter(torch.FloatTensor([configs.w_lin] * configs.enc_in), requires_grad=True)
        self.revin_layer = RevIN(configs.enc_in)
        self.log_prem = True

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, phase,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, ):

        x_enc = self.revin_layer(x_enc, 'norm')  # 归一化
        B, L, N = x_enc.shape
        # enc_out = x_enc.permute(0, 2, 1)  # [B, N, L]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, N, L, d_model]

        # 优化的日志记录方式
        if self.log_prem:
            logger.info(f"Input shape - B: {B}, L: {L}, N: {N}")
            logger.info(f"Encoded output shape: {enc_out.shape}")
        if self.use_ME:
            channel_emb = self.channel_embedding.expand(enc_out.shape[0], N, -1)  # [B, N, d_model]
            phase_emb = self.phase_embedding(phase.view(-1, 1).expand(B, N))  # [B, N, d_model]

            enc_out = enc_out[:, :N, :] + channel_emb + phase_emb

            if self.log_prem:
                logger.info(f"Channel embedding shape: {channel_emb.shape}")
                logger.info(f"Phase embedding shape: {phase_emb.shape}")
                logger.info(f"Final encoded output shape: {enc_out.shape}")

        enc_orgin = enc_out  # [B, N, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        if self.log_prem:
            logger.info(f"Encoder output shape: {enc_out.shape}")  # [B, N, d_model]
        # 残差投影
        dec_out = self.projector(enc_out + enc_orgin).permute(0, 2, 1)[:, :, :N]

        if self.log_prem:
            logger.info(f"Decoder output shape: {dec_out.shape}")  # [B, pre_L, N]
        # 线性分支
        if self.use_L:
            x = x_enc.permute(0, 2, 1)
            x3 = self.Linear(x)
            x3 = self.GeLU(x3)
            x3 = self.Hidden1(x3)
            linear_out = x3.permute(0, 2, 1)
            if self.use_T:
                dec_out = self.revin_layer(dec_out[:, -self.pred_len:, :] + self.w_dec * linear_out, 'denorm')  # 混合输出
            else:
                dec_out = self.revin_layer(linear_out, 'denorm')  #只有线性分支输出

            if self.log_prem:
                logger.info(f"Linear output shape: {linear_out.shape}")
                logger.info(f"Final decoder output shape: {dec_out.shape}")
        else:
            dec_out = self.revin_layer(dec_out[:, -self.pred_len:, :], 'denorm')  #只有Transformer分支输出
        # 只在第一次前向传播时记录日志
        self.log_prem = False

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out

    def moving_avg(self, x):
        """x: [B, L, N]"""
        B, L, N = x.shape
        x = x.permute(0, 2, 1).contiguous()  # [B, N, L]
        x = x.view(B * N, L)
        pad_len = self.kernel_size - 1
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (pad_len, 0), mode='replicate')
        avg = torch.nn.functional.avg_pool1d(x, kernel_size=self.kernel_size, stride=1)
        avg = avg.view(B, N, -1).permute(0, 2, 1)  # [B, L_out, N]
        if avg.shape[1] != L:
            avg = torch.nn.functional.interpolate(avg.permute(0, 2, 1), size=L, mode='linear', align_corners=False)
            avg = avg.permute(0, 2, 1)
        return avg
