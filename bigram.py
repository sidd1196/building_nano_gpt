
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from GLU import TransformerBlock

from SparseAttention import SparseAttention

from Linformer import LinformerSelfAttention  # Import Linformer (if using Linformer)
from Performer import PerformerSelfAttention  # Import Performer (if using Performer)

# Realted to tensorboard for logging
from torch.utils.tensorboard import SummaryWriter


# python environment = nano_gpt_env

# Heyperparameters after scaling 

# max_iters = 5000
# eval_interval = 500
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# learning_rate = 3e-4
# n_embd = 384
# n_head = 6
# n_layer = 6
# dropout = 0.2


# Hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 50
eval_interval = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
learning_rate = 1e-3
# n_embd = 36
# n_head = 6
n_embd = 72
n_head = 6

n_layer = 6
dropout = 0.1


torch.manual_seed(1337)

# print("imported libraries")


# Hyperparameters for Applying different Attention Mechanisms
use_glu = False  # Add this flag to choose whether to use GLU TransformerBlock
use_sparseattention = False  # Set to False to disable Sparse Attention
use_linformer = False  # Set to True if using Linformer
use_performer = False  # Set to True if using Performer


with open('input.txt', 'r') as file:
    text = file.read()

# Here we will create a dictionary of characters to integers
chars = sorted(list(set(text)))
vocab_size = len(chars)



print("================================================")

# Create a mapping from characters to integers
stoi = { ch:i for i, ch in enumerate(chars) } # Maps characters to integers 
itos = { i:ch for i, ch in enumerate(chars) } # Maps characters to integers 
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# print(f"stoi: {stoi}")
# print(f"itos: {itos}")

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long) # The text is converted into numbers 
# Converts the list of integers (output from encode) into a PyTorch tensor, which is the core data structure for PyTorch.
# PyTorch tensors are similar to NumPy arrays, but they are optimized for GPU acceleration and deep learning operations.

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading

# Logging the performance of models

from logger import ModelLogger

# Initialize the logger
logger = ModelLogger()

# Log the hyperparameters
hyperparameters = {
    'batch_size': batch_size,
    'block_size': block_size,
    'learning_rate': learning_rate,
    'n_embd': n_embd,
    'n_head': n_head,
    'n_layer': n_layer,
    'dropout': dropout
}
logger.log_hyperparameters(**hyperparameters)


def get_batch(split):
    data = train_data if split == 'train' else val_data
    random_indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in random_indices])
    y = torch.stack([data[i+1:i+block_size+1] for i in random_indices])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # this decorator is used to tell PyTorch that the function should not track gradients  
# No intedent of backpropogation to make more efficient

def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # B, T, C
        q = self.query(x) # B, T, C
        # Compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, 16) @ (B, 16, T) ----> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # B, T, T
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # Weighted aggregation of the values
        v = self.value(x) # B, T, C
        out = wei @ v # B, T, T @ B, T, C ----> B, T, C
        return out
    


class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size): 
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  # Correct attribute assignment
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) 
        # out = self.proj(out)
        out = self.dropout(self.proj(out))

        return out
class FeedForward(nn.Module):
    """ A simple layer followed by non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: Communication followed bycomputation """

    def __init__(self, n_embd, n_head, use_glu=False):
        super().__init__()

        if use_glu:
            self.sa = TransformerBlock(n_embd, n_head, glu_dim=n_embd)
        else:
            head_size = n_embd // n_head
            self.sa = MultiHeadAttention(n_head, head_size)

        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x): 
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# Simple Bigram Language Model

class BigramLanguageModel(nn.Module):

    def __init__(self, use_glu=False, use_sparseattention=False, use_linformer=False, use_performer=False):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Define attention type
        if use_sparseattention:
            self.attention = SparseAttention(embed_size=n_embd, num_heads=n_head, dropout=dropout, seq_length=block_size)
        elif use_linformer:
            self.attention = LinformerSelfAttention(n_embd, n_head, seq_len=block_size)
        elif use_performer:
            self.attention = PerformerSelfAttention(n_embd, n_head)
        else:
            self.attention = MultiHeadAttention(num_heads=n_head, head_size=n_embd // n_head)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head, use_glu=use_glu) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  
        x = tok_emb + pos_emb  
        
        x = self.blocks(x)  

        if use_sparseattention or use_linformer or use_performer:
            x = self.attention(x)  

        x = self.ln_f(x)  
        logits = self.lm_head(x)  

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, idx, max_new_tokens):
      # idx is the (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block size tokens 
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    

# Here starts the code for train.py in the github


model = BigramLanguageModel(use_glu=use_glu, use_sparseattention=use_sparseattention, use_linformer=use_linformer, 
                            use_performer=use_performer)

# Move model to bfloat16 to reduce memory usage and improve training efficiency
model.to(torch.bfloat16)


# Initializing for recording the losses and gradients - Tendsorboard
writer = SummaryWriter(log_dir="logs")

# create a PyTorch optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(max_iters):

    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # Log the losses
        logger.log_loss(losses['train'], losses['val'])

                
        # Log losses to TensorBoard
        writer.add_scalars("Loss", {'train': losses['train'], 'val': losses['val']}, i)


    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    # Log gradients and weights
    logger.log_gradients(model)
    logger.log_weights(model)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()  #  Backpropagation (calculating gradients)
    optimizer.step() #  Updating the model's parameters using those gradients

    # Log model weights and gradients TensorBoard
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            writer.add_histogram(f"weights/{name}", param, i)
            writer.add_histogram(f"gradients/{name}", param.grad, i)

# Close the logger at the end of training
logger.close()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))




