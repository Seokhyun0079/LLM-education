import torch
from torch.utils.data import DataLoader
import tiktoken
from torch.utils.data import Dataset

input_ids = torch.tensor([2,3,5,1])
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer.weight)

print(embedding_layer(input_ids))

vocab_size = 50257
ouput_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, ouput_dim)

class GpTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

max_length = 4
def craete_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GpTDatasetV1(txt, tokenizer, max_length, stride)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return data_loader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
dataloader = craete_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Toekn Ids:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, ouput_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings

print(input_embeddings.shape)








