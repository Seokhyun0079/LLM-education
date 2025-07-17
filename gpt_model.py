from torch import nn
from transformer_block import TransformerBlock
from dummy_gpt_model import LayerNorm
import torch

class GPTModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
    self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
    self.drop_emb = nn.Dropout(cfg["drop_rate"])

    self.trf_blcoks = nn.Sequential(*[
      *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
    ])

    self.final_norm = LayerNorm(cfg["emb_dim"])
    self.out_head = nn.Linear(
      cfg["emb_dim"], cfg["vocab_size"], bias=False
    )
  def forward(self, in_index):
    batch_size, seq_len = in_index.shape
    tok_embeds = self.tok_emb(in_index)

    pos_embes = self.post_emb(
      torch.arange(seq_len, device= in_index.device)
    )
    x = tok_embeds + pos_embes
    x = self.drop_emb(x)
    x = self.trf_blcoks(x)
    x = self.final_norm(x)
    logits = self.out_head(x)
    return logits

