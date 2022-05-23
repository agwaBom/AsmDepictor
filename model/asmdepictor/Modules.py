import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        round_param = 5
        k = k.transpose(2, 3)
        q = q / self.temperature
        attn = torch.matmul(q, k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Unique-Softmax
        maxes = torch.max(attn, dim=-1, keepdim=True)[0]
        x_exp = torch.exp(attn-maxes)
        x_exp = torch.round(x_exp * 10**round_param) / (10**round_param)
        y,_ = x_exp.sort(dim=-1)
        y[:,:,:, 1:] *= ((y[:,:,:, 1:] - y[:,:,:, :-1]) !=0).long()
        x_exp_sum = torch.sum(y, dim=-1, keepdim=True)
        attn = self.dropout(x_exp/x_exp_sum)
        
        output = torch.matmul(attn, v)

        return output, attn