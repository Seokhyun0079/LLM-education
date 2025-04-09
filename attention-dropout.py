import torch
from selfAttetionClass import SelfAttention_v2
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6,6)
print(dropout(example))
torch.manual_seed(123)
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

sa_v2 = SelfAttention_v2(d_in=3, d_out=2)
queries = sa_v2.w_query(inputs)
keys = sa_v2.w_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
print(attn_weights)
print(dropout(attn_weights))


