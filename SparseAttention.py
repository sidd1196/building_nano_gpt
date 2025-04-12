import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAttention(nn.Module):
    def __init__(self, embed_size, num_heads, seq_length, window_size=256, sparsity_pattern='sliding_window', dropout=0.1):
        super(SparseAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.window_size = window_size
        self.sparsity_pattern = sparsity_pattern
        
        # Dropout only on attention weights
        self.attn_dropout = nn.Dropout(dropout)
        
        # Define Q, K, V linear transformations
        self.q_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)
        
        # Define Output linear transformation
        self.out_linear = nn.Linear(embed_size, embed_size)
        
        # Set up sparsity pattern
        self._setup_sparsity_pattern()

    def _setup_sparsity_pattern(self):
        if self.sparsity_pattern == 'sliding_window':
            self.attention_mask = self._create_sliding_window_mask()

    def _create_sliding_window_mask(self):
        """Creates a sliding window mask to restrict attention to local neighbors."""
        mask = torch.zeros(self.seq_length, self.seq_length)
        for i in range(self.seq_length):
            for j in range(max(0, i - self.window_size), min(self.seq_length, i + self.window_size)):
                mask[i, j] = 1
        return mask

    def forward(self, x):
        batch_size = x.size(0)

        # Linear projections
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        # Debug prints to check dimensions
        print(f"Q shape: {Q.shape}")
        print(f"K shape: {K.shape}")
        print(f"V shape: {V.shape}")
        print(f"batch_size: {batch_size}, num_heads: {self.num_heads}, seq_length: {self.seq_length}, embed_size: {self.embed_size}")


        # Reshape for multi-head attention
        seq_length = Q.shape[1]  # Use actual sequence length from input
        Q = Q.view(batch_size, seq_length, self.num_heads, self.embed_size // self.num_heads).permute(0, 2, 1, 3)
        K = K.view(batch_size, self.num_heads, self.seq_length, self.embed_size // self.num_heads)
        print(f"Reshaped K shape: {K.shape}")
        V = V.view(batch_size, self.seq_length, self.num_heads, self.embed_size // self.num_heads).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.embed_size // self.num_heads, dtype=torch.float32))
        
        # Apply mask for sparse attention
        scores = scores.masked_fill(self.attention_mask == 0, float('-inf'))

        # Apply dropout to attention scores before softmax
        scores = self.attn_dropout(scores)

        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Apply dropout to attention weights
        attention_weights = self.attn_dropout(attention_weights)

        # Apply attention weights to values
        attention_output = torch.matmul(attention_weights, V)

        # Reshape output back
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, self.seq_length, self.embed_size)

        # Final output projection
        output = self.out_linear(attention_output)
        return output
