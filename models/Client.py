import logging
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mpmath import phase

from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.Autoformer_EncDec import moving_avg
from layers.SelfAttention_Family import FullAttention, AttentionLayer, ProbAttention, DSAttention
from layers.Embed import DataEmbedding, DataEmbedding_inverted
import numpy as np
from layers.RevIN import RevIN
# 创建 logs 目录（如果不存在）
os.makedirs("logs", exist_ok=True)

# 配置日志
logging.basicConfig(
    filename='logs/experiment.log',          # 日志文件路径
    filemode='a',                            # 追加模式（'w' 会覆盖）
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO                      # 只记录 INFO 及以上级别
)
logger = logging.getLogger()
warnings.filterwarnings('ignore')


class CrossModalAttentionFusion(nn.Module):
    """交叉注意力融合模块"""

    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # 三种模态间的交叉注意力
        self.cross_attn_c2p = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )
        self.cross_attn_p2j = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )
        self.cross_attn_j2c = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )

        # 自注意力用于最终融合
        self.self_attn = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )

        # 残差连接和层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm_final = nn.LayerNorm(d_model)

        # 门控机制（可选）
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Sigmoid()
        )

        # 输出投影
        self.output_proj = nn.Linear(d_model * 3, d_model)

    def forward(self, channel_emb, phase_emb, joint_emb):
        """
        channel_emb: [B, N, D]
        phase_emb: [B, N, D]
        joint_emb: [B, N, D]
        返回: [B, N, D]
        """
        B, N, D = channel_emb.shape

        # 1. 交叉注意力：通道→相位
        channel_to_phase, _ = self.cross_attn_c2p(
            phase_emb,  # Query
            channel_emb,  # Key
            channel_emb  # Value
        )
        channel_to_phase = self.norm1(channel_emb + channel_to_phase)

        # 2. 交叉注意力：相位→联合
        phase_to_joint, _ = self.cross_attn_p2j(
            joint_emb,
            phase_emb,
            phase_emb
        )
        phase_to_joint = self.norm2(phase_emb + phase_to_joint)

        # 3. 交叉注意力：联合→通道
        joint_to_channel, _ = self.cross_attn_j2c(
            channel_emb,
            joint_emb,
            joint_emb
        )
        joint_to_channel = self.norm3(joint_emb + joint_to_channel)

        # 4. 拼接三种增强后的表示
        fused = torch.cat([channel_to_phase, phase_to_joint, joint_to_channel], dim=-1)

        # 5. 门控加权（可选）
        gate_weights = self.gate(fused)  # [B, N, D]
        # 这里可以按维度分割gate_weights分别加权三个分量

        # 6. 自注意力进一步融合
        fused_proj = self.output_proj(fused)  # [B, N, D]
        fused_final, _ = self.self_attn(
            fused_proj,
            fused_proj,
            fused_proj
        )
        fused_final = self.norm_final(fused_proj + fused_final)

        return fused_final

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # print(configs.d_model)
        #todo: d_model 最好是256但是作者给直接和seq_len一样了
        # configs.d_model = configs.seq_len

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
        #通道嵌入
        self.channel_embedding = nn.Parameter(torch.zeros(configs.enc_in, configs.d_model))
        self.phase_embedding = nn.Embedding(self.cycle_len,configs.d_model)
        nn.init.xavier_normal(self.phase_embedding.weight)
        self.joint_embedding = nn.Embedding(self.cycle_len,self.enc_in*self.d_model)
        nn.init.xavier_normal(self.joint_embedding.weight)
        nn.init.xavier_normal(self.channel_embedding)

        self.projector = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model * 2),
            nn.GELU(),
            nn.Dropout(configs.output_proj_dropout),
            nn.Linear(configs.d_model * 2, configs.d_model * 4),
            nn.GELU(),
            nn.Dropout(configs.output_proj_dropout),
            nn.Linear(configs.d_model * 4, configs.pred_len),
        )
        self.Linear = nn.Linear(self.seq_len, self.seq_len)
        self.GeLU = nn.GELU()
        self.Hidden1 = nn.Linear(self.seq_len, self.pred_len)
        self.w_dec = torch.nn.Parameter(torch.FloatTensor([configs.w_lin]*configs.enc_in),requires_grad=True)
        self.revin_layer = RevIN(configs.enc_in)
        self.log_prem = True
        self.use_L = configs.use_L
        self.cross_attention = CrossModalAttentionFusion(
            d_model=configs.d_model,
            n_heads=configs.n_heads,  # 使用相同的头数
            dropout=configs.dropout
        )


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,phase,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,):

        x_enc = self.revin_layer(x_enc, 'norm')#归一化
        B,L,N = x_enc.shape


        # enc_out = x_enc.permute(0, 2, 1)#[batch_size,enc_in,seq_len]，[32,7,96]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)#[batch_size,enc_in,seq_len]，[32,7,96]
        if self.log_prem:
            logger.info("B,L,N", B, L, N)
            logger.info("enc_out", enc_out.shape)

        channel_emb = self.channel_embedding.expand(enc_out.shape[0], N, -1)

        phase_emb = self.phase_embedding(phase.view(-1, 1).expand(B, N))
        joint_emb = self.joint_embedding(phase).reshape(B, self.enc_in, self.d_model)
        # enc_out = enc_out[:, :N, :] + channel_emb + phase_emb + joint_emb  # [B,N,],[32, 7, 256]
        fused_emb = self.cross_attention(channel_emb, phase_emb, joint_emb)
        enc_out = enc_out[:, :N, :] + fused_emb
        if self.log_prem:
            logger.info("channel_emb", channel_emb.shape)
            logger.info("phase_emb", phase_emb.shape)
            logger.info("joint_emb", joint_emb.shape)
            logger.info("enc_out", enc_out.shape)
        enc_orgin = enc_out
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)#[batch_size,enc_in,seq_len],[32, 7, 96]

        dec_out = self.projector(enc_out + enc_orgin).permute(0, 2, 1)[:, :, :N]
        if self.log_prem:
            logger.info("dec_out", dec_out.shape)
        if self.use_L:
            x = x_enc.permute(0,2,1)
            x3 = self.Linear(x)
            x3 = self.GeLU(x3)
            x3 = self.Hidden1(x3)
            linear_out= x3.permute(0,2,1)
            dec_out = self.revin_layer(dec_out[:, -self.pred_len:, :]+self.w_dec*linear_out, 'denorm')#[batch_size,seq_len,enc_in][32, 96, 7]
            if self.log_prem:
                logger.info("dec_out", dec_out.shape)
                logger.info("linear_out", linear_out.shape)
        else:
            dec_out = self.revin_layer(dec_out[:, -self.pred_len:, :], 'denorm')#[batch_size,seq_len,enc_in][32, 96, 7]
            if self.log_prem:
                logger.info("dec_out", dec_out.shape)
        self.log_prem = False
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out