import torch
import torch.nn as nn
from GELU import GELU

GPT_CONFIG_124M = {
  "vocab_size": 50257,
  "context_length": 1024,
  "emb_dim": 768,
  "n_heads": 12,
  "n_layers" : 12,
  "drop_rate" : 0.1,
  "qkv_bias" : False
}


class FeedForward(nn.module):
  def __init__(self, cfg):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
      GELU(),
      nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
    )
  def forwad(self, x):
    return self.layers(x)

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.randn([2, 3, 768])
out = ffn(x)
print(out.shape)