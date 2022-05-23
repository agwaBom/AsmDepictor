import torch.nn as nn
import torch
from model.asmdepictor.SubLayers import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, n_position, d_word_vec, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        self.n_position = n_position-3
        self.position_emb = nn.Embedding(n_position, d_word_vec)

        # rezero implementation
        self.reweight = nn.Parameter(torch.Tensor([0]))
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, enc_input, slf_attn_mask=None):
        rezero = False

        a = torch.arange(self.n_position, device=enc_input.device).expand((1, -1))
        position_emb = self.position_emb(a)        
        enc_input += position_emb

        if rezero:
            tmp_enc_input = enc_input

            enc_output, enc_slf_attn = self.slf_attn(
                enc_input, enc_input, enc_input, is_encoder=True, mask=slf_attn_mask)
            
            enc_output = enc_output * self.reweight
            enc_output = tmp_enc_input + self.dropout1(enc_output)

            tmp_enc_output = enc_output
            enc_output = self.pos_ffn(tmp_enc_output)

            enc_output = enc_output * self.reweight
            enc_output = tmp_enc_output + self.dropout2(enc_output)

        else:
            enc_output, enc_slf_attn = self.slf_attn(
                enc_input, enc_input, enc_input, is_encoder=True, mask=slf_attn_mask)
            enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        # rezero implementation
        self.reweight = nn.Parameter(torch.Tensor([0]))
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_enc_attn_mask=None
        rezero = False

        if rezero:
            tmp_dec_input = dec_input
            dec_output, dec_slf_attn = self.slf_attn(
                dec_input, dec_input, dec_input, mask=slf_attn_mask)
            dec_output = dec_output * self.reweight
            dec_output = tmp_dec_input + self.dropout1(dec_output)
            
            tmp_dec_output = dec_output
            dec_output, dec_enc_attn = self.enc_attn(
                dec_output, enc_output, enc_output, dec_enc_conn=True, mask=dec_enc_attn_mask)
            dec_output = dec_output * self.reweight
            dec_output = tmp_dec_output + dec_output

            tmp_dec_output = dec_output
            dec_output = self.pos_ffn(dec_output)
            dec_output = dec_output * self.reweight
            dec_output = tmp_dec_output + dec_output
        else:
            dec_output, dec_slf_attn = self.slf_attn(
                dec_input, dec_input, dec_input, mask=slf_attn_mask)
            dec_output, dec_enc_attn = self.enc_attn(
                dec_output, enc_output, enc_output, dec_enc_conn=True, mask=dec_enc_attn_mask)
            dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn