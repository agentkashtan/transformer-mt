import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch


class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, seq_length, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        pe = torch.zeros(self.seq_length, d_model)
        position = torch.arange(0, self.seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, features, eps=10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=2, keepdim=True)
        std = ((x - mean) ** 2).mean(dim=2, keepdim=True)
        return self.alpha * (x - mean) / torch.sqrt(std + self.eps) + self.bias

class FFNN(nn.Module):
    def __init__(self, d_model, d_internal, dropout):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_internal)
        self.layer2 = nn.Linear(d_internal, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        return self.layer2(self.dropout(self.relu(self.layer1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        #(batch, h, seq, d_k)
        d_k = query.shape[-1]
        scores = (query @ key.transpose(2, 3)) / math.sqrt(d_k)
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        weights = F.softmax(scores, dim=3)
        if dropout is not None:
            weights = dropout(weights)
        return weights @ value

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_model // self.h).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_model // self.h).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_model // self.h).transpose(1, 2)
        multihead_attention = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        multihead_attention = multihead_attention.transpose(1, 2).contiguous().view(query.shape[0], query.shape[2], self.d_model)
        return self.w_o(multihead_attention)

class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))



class EncoderBlock(nn.Module):
    def __init__(self, d_model, dropout, h, d_internal):
        super().__init__()
        self.multihead_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])
        self.ffnn = FFNN(d_model, d_internal, dropout)

    def forward(self, x, mask):
        x = self.residual_connections[0](x, lambda x: self.multihead_attention(x, x, x, mask))
        x = self.residual_connections[1](x, self.ffnn)
        return x

class Encoder(nn.Module):
    def __init__(self, encoder_num, d_model, dropout, h, d_internal):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model, dropout, h, d_internal) for _ in range(encoder_num)])
        self.norm = LayerNormalization(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x , mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, dropout, h, d_internal):
        super().__init__()
        self.multihead_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        self.residual_connections = nn.ModuleList(ResidualConnection(d_model, dropout) for _ in range(3))
        self.ffnn = FFNN(d_model, d_internal, dropout)
        self.multihead_cross_attention = MultiHeadAttentionBlock(d_model, h, dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.multihead_self_attention(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.multihead_cross_attention(x, encoder_output, encoder_output, src_mask))
        return self.residual_connections[2](x, self.ffnn)

class Decoder(nn.Module):
    def __init__(self, decoder_num, d_model, dropout, h, d_internal):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model, dropout, h, d_internal) for  _ in range(decoder_num)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.layer(x)

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_seq_length,
        tgt_seq_length,
        encoder_num=6,
        decoder_num=6,
        d_model=512,
        dropout=0.1,
        h=8,
        d_internal=2048
        ):
        super().__init__()
        self.encoder = Encoder(encoder_num, d_model, dropout, h, d_internal)
        self.decoder = Decoder(encoder_num, d_model, dropout, h, d_internal)
        self.src_embed = InputEmbedding(d_model, src_vocab_size)
        self.tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
        self.src_pos = PositionalEmbedding(d_model, src_seq_length, dropout)
        self.tgt_pos = PositionalEmbedding(d_model, tgt_seq_length, dropout)
        self.proj = ProjectionLayer(d_model, tgt_vocab_size)

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        tgt = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return tgt

    def project(self, x):
        return self.proj(x)

def c_mask(size):
    mask = torch.triu(torch.ones(size, size), 1).type(torch.int)
    return mask == 0

