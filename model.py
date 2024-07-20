batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6 #each head is 64 dimensional
n_layer = 6
dropout = 0.2

class LanguageModel(nn.Module):
    '''
    Define the base language model with all subcomponents
    '''
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        #make number of blocks and number of heads variable
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) 

        
    def forward(self, idx, targets):
    
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #(T,C)
        x = tok_emb+pos_emb
        x = self.blocks(x)
        x = self.ln_f(x) #pass x through layernorm
        logits = self.lm_head(x)
        B,T,C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx
      
class FeedForward(nn.Module):
  '''
  Define a feed forward module in the language model
  '''
  def __init__(self, n_embd):
      super().__init__()
      self.net = nn.Sequential(
          nn.Linear(n_embd, n_embd),
          nn.ReLU(),
          nn.Linear(n_embd, n_embd)
          nn.Dropout(dropout) #add dropout
      )
  def forward(self, x):
      return self.net(x)

class MultiHeadAttention(nn.Module):
  '''
  Define a multi-headed attention module in the language model
  '''
  
  def __init__(self, num_heads, head_size):
      super().__init__()
      self.heads = nn.ModuleList([Head(head_size) for i in range(num_heads)])
      self.proj = nn.Linear(n_embd, n_embd)

      #add dropout
      self.dropout = nn.Dropout(dropout)
  def forward(self, x, targets):
      out = torch.cat([h(x) for h in self.heads], dim = -1)
      #add dropout
      out = self.dropout(self.proj(out))
      return out
class Head(nn.Module):
  '''
  Define a single attention head
  '''
  def __init__(self, head_size):
      super().__init__()
      self.key = nn.Linear(vocab_size, head_size, bias = False)
      self.query = nn.Linear(vocab_size, head_size, bias = False)
      self.value = nn.Linear(vocab_size, head_size, bias = False)
      self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

      #add dropout
      self.dropout = nn.Dropout(dropout)

  def forward(self, x, targets):
      B, T, C = x.shape
      k = self.key(idx)
      q = self.query(idx)
      v = self.value(idx)
      wei = q @ k.transpose(-2, -1) * C**-0.5

      wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
      wei = F.softmax(wei, dim = -1)

      #apply dropout to randomly prevent some nodes from communicating
      wei = self.dropout(wei)

      out = wei @ v
      return out
class Block(nn.Module):
  '''
  Define a single block
  '''
  def __init__(self, n_embd, n_head):
      super().__init__()
      head_size = n_embd//n_head
      self.sa = MultiHeadAttention(n_head, head_size)
      self.ffwd = FeedForward(n_embd)
      #initialize layernorms
      self.ln1 = nn.LayerNorm(n_embd)
      self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
      x = x+ self.sa(self.ln1(x)) #apply layernorms before feeding x into the attention heads
      x = x+ self.ffwd(self.ln2(x))
      return x
