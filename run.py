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
from .data import TextData
from .model import LanguageModel
data = TextData()
m = LanguageModel(data.vocab_size)
#used because converges faster in some situations
optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

def train():
  for steps in range(max_iters):
      #get train data using our batch function
      xb, yb = data.get_batch('train')
      #eval loss and update model
      logits, loss = m(xb, yb)
      optimizer.zero_grad(set_to_none = True)
      loss.backward()
      optimizer.step()
      if steps%500 == 0:
        with model.eval():
          xb, yb = data.get_batch('val')
          logits, loss = m(xb, yb)
          print(f"step {steps} loss: {loss}")
          

print(decode(m.generate(idx = torch.zeros((1,1), dtype = torch.log), max_new_tokens = 500)[0].tolist()))
