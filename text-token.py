import urllib.request
url = ("https://raw.githubusercontent.com/rasbt/" "LLMs-from-scratch/main/ch02/01_main-chapter-code/" "the-verdict.txt")
file_path = "the-verdict.txt"

urllib.request.urlretrieve(url, file_path)

with open(file_path, "r") as f:
    raw_text = f.read()
    
print("Total number of characters in the text: ", len(raw_text))
print(raw_text[:99])

import re

text = "Hello, World!. This, is a test."
result = re.split(r'([,.]|\s)',text)
print(result)

result = [item for item in result if item.strip()]
print(result)

text ="Hello, world. Is this-- a test?"
result = re.split(r'(,[.:;?_!"()\']|--|\s)', text)
result = [item for item in result if item.strip()]
print(result)

preprocessed1 = re.split(r'(,[.:;?_!"()\']|--|\s)', raw_text)
preprocessed1 = [item for item in preprocessed1 if item.strip()]

# print(preprocessed[:30])

all_words = sorted(set(preprocessed1))
# vocab_size = len(all_words)
# print(f"Vocabulary size: {vocab_size}")

vocab = {token: integer for integer, token in enumerate(all_words)}
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i > 50:
#         break

class SimpleTokeninzerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i : s for s, i in vocab.items()}       
    def encode(self, text):
        preprocessed = re.split(r'(,[.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([m.?!"()\'])', r'\1 ', text)
        return text

tokenizer = SimpleTokeninzerV1(vocab)
text = """"It's the last he painted, you know,"
        Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

decoded = tokenizer.decode(ids)
print(decoded)

all_tokens = sorted(list(set(preprocessed1)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}

print(len(vocab))

for i, item in enumerate(vocab.items()):
    print(item)

class SimpleTokeninzerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i : s for s, i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'(,[.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1 ', text)
        return text

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = "<|startoftext|>".join((text1, text2))

tokenizer = SimpleTokeninzerV2(vocab)
print(tokenizer.encode(text))
decoded = tokenizer.decode(tokenizer.encode(text))
print(decoded)



