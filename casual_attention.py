import torch
import torch.nn as nn

class CasualAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
    super().__init__()
    self.d_out = d_out
    self.context_length = context_length
    self.dropout = dropout
    self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    self.dropout = nn.Dropout(dropout)
    self.register_buffer('mask', torch.tril(torch.ones(context_length, context_length), diagonal=1))

  def forward(self, x):
    b, num_tokens, d_in = x.shape
    keys = self.w_key(x)
    queries = self.w_query(x)
    values = self.w_value(x)

    attn_scores = queries @ keys.transpose(1, 2)    
    attn_scores.masked_fill(
      self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
    )
    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim= -1)
    attn_weights = self.dropout(attn_weights)

    context_vec = attn_weights @ values
    return context_vec
