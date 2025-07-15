import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    assert(d_out % num_heads == 0), "d_out must be divisible by num_heads"
    
    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    print(self.W_query)
    print(self.W_key)
    print(self.W_value)

    self.out_proj = nn.Linear(d_out, d_out)

    self.dropout = nn.Dropout(dropout)
    self.register_buffer(
      "mask",
      torch.triu(torch.ones(context_length, context_length), diagonal=1)
    )    

  def forward(self, x):
    b, num_tokens, d_in = x.shape

    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)

    print('keys',keys)
    print('queries',queries)
    print('values',values)

    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
    values = values.view(b, num_tokens, self.num_heads, self.head_dim)
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

    print('after view')
    print('keys',keys)
    print('values',values)
    print('queries',queries)
    
    keys = keys.transpose(1, 2)
    queries = queries.transpose(1, 2)
    values = values.transpose(1, 2)

    print('after transpose')
    print('keys',keys)
    print('values',values)
    print('queries',queries)

    attn_scores = queries @ keys.transpose(2, 3)

    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

    attn_scores.masked_fill_(mask_bool, -torch.inf)

    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)

    context_vec = (attn_weights @ values).transpose(1, 2)
    context_vec = context_vec.contiguous().view( 
      b, num_tokens, self.d_out
    )

    context_vec = self.out_proj(context_vec)

    return context_vec



# mha = MultiHeadAttention(
#   d_in=2,
#   d_out=4,
#   context_length=2,
#   dropout=0.1,
#   num_heads=2
# )

# x = torch.randn(1, 2, 2)

# print(x)

# print(mha(x))


# a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
#                     [0.8993, 0.0390, 0.9268, 0.7388],
#                     [0.7173, 0.7058, 0.9156, 0.4340]],
                    
#                     [[0.0772, 0.3565, 0.1479, 0.5331],
#                     [0.4066, 0.2318, 0.4545, 0.9737],
#                     [0.4606, 0.5159, 0.4220, 0.5786]]]])
# print('a.transpose(2, 3)')
# print(a.transpose(2, 3))
# print('a @ a.transpose(2, 3)')
# print(a @ a.transpose(2, 3))
# #[[0.2745, 0.6584, 0.2775, 0.8573],
# # [0.8993, 0.0390, 0.9268, 0.7388],
# # [0.7173, 0.7058, 0.9156, 0.4340]]
# first_head = a[0, 0,:, :]
# first_res = first_head @ first_head.T
# print('first_head:\n', first_head)

# second_head = a[0, 1,:, :]
# second_res = second_head @ second_head.T
# print('second_head:\n', second_head)

# print('first_res:\n', first_res)
# print('second_res:\n', second_res)

