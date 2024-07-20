import torch

class TextData:
  def __init__(self, block_size, batch_size, split_size=0.8):
    !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open('input.txt', 'r', encoding='utf-8') as f:
      self.text = f.read()
    self.chars = sorted(list(set(self.text)))
    self.vocab_size = len(chars)
    
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    self.encode = lambda s: [stoi[c] for c in s]
    self.decode = lambda l: ''.join([itos[i] for i in l])
    data = torch.tensor(self.encode(self.text), dtype=torch.long)
    n = int(split_size*len(data))
    self.train_data = data[:n]
    self.val_data = data[n:]
    
  def get_batch(split):
    data = self.train_data if split == 'train' else self.val_data
    ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
    x = torch.stack([data[i:i+self.block_size] for i in ix])
    y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
    return x, y
