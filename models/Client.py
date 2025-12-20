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
        self.use_ME = configs.use_ME  # 使用多嵌入
        self.use_L = configs.use_L  # 使用使用线性分支
        self.use_R = configs.use_R  # 使用归一化
        self.seq_len = configs.seq_len
        # self.use_norm = configs.use_norm
        self.d_model = configs.d_model
        self.cycle_len = configs.cycle
        self.enc_in = configs.enc_in
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
            nn.init.xavier_normal_(self.channel_embedding)
            self.emb_norm = nn.LayerNorm(configs.d_model)
            self.emb_drop = nn.Dropout(configs.dropout)
            self.a_c = nn.Parameter(torch.tensor(0.1))
            self.a_p = nn.Parameter(torch.tensor(0.1))
            self.a_j = nn.Parameter(torch.tensor(0.1))

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
            self.w_dec_raw = nn.Parameter(torch.full((configs.enc_in,), -2.0))
        if self.use_R:
            self.revin_layer = RevIN(configs.enc_in)
        self.log_prem = True


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, phase,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, ):
        if self.use_R:
            x_enc = self.revin_layer(x_enc, 'norm')  # 归一化
        else:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
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
            joint_emb = self.joint_embedding(phase).reshape(B, self.enc_in, self.d_model)  # [B, N, d_model]
            enc_out = self.emb_drop(self.emb_norm(
                enc_out[:, :N, :] + self.a_c * channel_emb + self.a_p * phase_emb + self.a_j * joint_emb
            ))

            if self.log_prem:
                logger.info(f"Channel embedding shape: {channel_emb.shape}")
                logger.info(f"Phase embedding shape: {phase_emb.shape}")
                logger.info(f"Joint embedding shape: {joint_emb.shape}")
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
            dec_out = self.revin_layer(dec_out[:, -self.pred_len:, :] + self.w_dec * linear_out, 'denorm')

        if self.use_R:
            dec_out = self.revin_layer(dec_out, 'norm')
        else:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))


        # 只在第一次前向传播时记录日志
        self.log_prem = False

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
