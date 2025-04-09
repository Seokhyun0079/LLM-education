import torch
import torch.nn as nn

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
class SelfAttention_v2(nn.Module):
  def __init__(self, d_in, d_out, qkv_bias=False):
    super().__init__()
    self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    print("self.w_query: \n", self.w_query)
    self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)

  def forward(self, x):
    keys = self.w_key(x)
    queries = self.w_query(x)
    print("queries: \n", queries)
    values = self.w_value(x)

    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(
      attn_scores / keys.shape[-1] ** 0.5, dim= -1
    )    
    context_vec = attn_weights @ values
    return context_vec
torch.manual_seed(123)
sa_v2 = SelfAttention_v2(d_in=3, d_out=2)
print(sa_v2(inputs))

queries = sa_v2.w_query(inputs)
keys = sa_v2.w_key(inputs)

attn_scores = queries @ keys.T
print("attn_scores: ", attn_scores)
print('keys.shape[-1]: ', keys.shape[-1])
print('keys.shape[-1] ** 0.5: ', keys.shape[-1] ** 0.5)
print('attn_scores / keys.shape[-1] ** 0.5: ', attn_scores / keys.shape[-1] ** 0.5)
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
print(attn_weights)

context_length = attn_scores.shape[0]
print("context_length: \n", context_length)
print("torch.ones(context_length, context_length): \n", torch.ones(context_length, context_length))
mask_simple = torch.tril(torch.ones(context_length, context_length))
print("mask_simple: \n", mask_simple)

masked_simple = attn_weights*mask_simple
print("masked_simple: \n", masked_simple)

row_sums = masked_simple.sum(dim=-1, keepdim=True)
print("row_sums: \n", row_sums)
masked_simple_norm = masked_simple / row_sums
print("masked_simple_norm: \n", masked_simple_norm)

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
print("mask: \n", mask)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print("masked: \n", masked)

attn_weights_masked = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
print("attn_weights_masked: \n", attn_weights_masked)


