import logging

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

logger = logging.getLogger(__name__)
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
        self.Linear = nn.Sequential()
        self.Linear.add_module('Linear',nn.Linear(configs.seq_len, self.pred_len))
        self.w_dec = torch.nn.Parameter(torch.FloatTensor([configs.w_lin]*configs.enc_in),requires_grad=True)
        self.revin_layer = RevIN(configs.enc_in)
        self.log_prem = True


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
        enc_out = enc_out[:, :N, :] + channel_emb + phase_emb + joint_emb  # [B,N,],[32, 7, 256]
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

        linear_out = self.Linear(x_enc.permute(0,2,1)).permute(0,2,1)#[batch_size,seq_len,enc_in][32, 96, 7]

        dec_out = self.revin_layer(dec_out[:, -self.pred_len:, :]+self.w_dec*linear_out, 'denorm')#[batch_size,seq_len,enc_in][32, 96, 7]
        if self.log_prem:
            logger.info("dec_out", dec_out.shape)
            logger.info("linear_out", linear_out.shape)
            self.log_prem = False
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out
