import torch.nn as nn
import torch

inputs = torch.tensor(
  [
    [0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66],# journey (x^2 )
    [0.57, 0.85, 0.64],#starts (x^3)
    [0.22, 0.58, 0.33],#with (x^4)
    [0.77, 0.25, 0.10],#one (x^5)
    [0.05, 0.80, 0.55] #step (x^6)
  ]
)
class SelfAttention_v1(nn.Module):
  def __init__(self, d_in, d_out):
    super().__init__()
    self.w_query = nn.Parameter(torch.randn(d_in, d_out))
    self.w_key = nn.Parameter(torch.randn(d_in, d_out))
    self.w_value = nn.Parameter(torch.randn(d_in, d_out))

  def forward(self, x):
    queries = x @ self.w_query
    keys = x @ self.w_key
    values = x @ self.w_value

    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(
      attn_scores / keys.shape[-1] ** 0.5, dim= -1
    )    
    context_vec = attn_weights @ values
    return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in=3, d_out=2)
print(sa_v1(inputs))