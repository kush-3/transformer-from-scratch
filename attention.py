"""
Transformer Attention Mechanisms - From Scratch Implementation

This module implements the core attention mechanisms and building blocks
of a transformer model using only NumPy. It includes:
- Scaled dot-product attention (both regular and causal)
- Multi-head attention mechanism
- Positional encoding using sinusoidal functions
- Feed-forward networks with ReLU activation
- Layer normalization for training stability
- Complete transformer block combining all components

The implementation follows the "Attention is All You Need" paper architecture.
"""

import numpy as np

def softmax(x):
    """
    Apply softmax activation function along the last dimension.
    
    Uses the numerically stable version by subtracting the maximum value
    to prevent overflow when exponentiating large numbers.
    
    Args:
        x (np.ndarray): Input tensor of any shape
        
    Returns:
        np.ndarray: Softmax probabilities (same shape as input)
    """
    # Subtract max for numerical stability (prevents exp overflow)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def self_attention(X, W_Q, W_K, W_V):
    """
    Implements scaled dot-product attention mechanism.
    
    This is the core attention function that computes how much each position
    in the sequence should attend to every other position. The scaling by
    sqrt(d_k) prevents the dot products from becoming too large.
    
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    
    Args:
        X (np.ndarray): Input embeddings of shape (seq_len, d_model)
        W_Q (np.ndarray): Query weight matrix of shape (d_model, d_k)
        W_K (np.ndarray): Key weight matrix of shape (d_model, d_k) 
        W_V (np.ndarray): Value weight matrix of shape (d_model, d_v)
        
    Returns:
        tuple: (output, attention_weights)
            - output: Attended values of shape (seq_len, d_v)
            - attention_weights: Attention scores of shape (seq_len, seq_len)
    """
    # Linear projections: transform input to queries, keys, and values
    Q = X @ W_Q  # Queries: what we're looking for
    K = X @ W_K  # Keys: what we're matching against
    V = X @ W_V  # Values: what we actually use in the output

    # Compute attention scores (how much each query attends to each key)
    scores = Q @ K.T  # Shape: (seq_len, seq_len)
    
    # Scale by sqrt(d_k) to prevent vanishing gradients with large dimensions
    d_k = K.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)
    
    # Convert scores to probabilities (each row sums to 1)
    attention_weights = softmax(scaled_scores)
    
    # Apply attention weights to values to get the final output
    output = attention_weights @ V
    
    return output, attention_weights


def multi_head_attention(x, n_heads, d_model, d_k, d_v):
    """
    Multi-Head Attention allows the model to jointly attend to information
    from different representation subspaces at different positions.
    
    Instead of performing a single attention function with d_model-dimensional
    keys, values and queries, we linearly project them h times with different,
    learned linear projections.
    
    Args:
        x (np.ndarray): Input of shape (seq_len, d_model)
        n_heads (int): Number of attention heads
        d_model (int): Model dimension
        d_k (int): Dimension of keys and queries per head
        d_v (int): Dimension of values per head
        
    Returns:
        np.ndarray: Multi-head attention output of shape (seq_len, d_model)
    """
    heads = []
    
    # Run attention in parallel for each head with different weight matrices
    for i in range(n_heads):
        # Each head gets its own random weight matrices
        # In practice, these would be learned parameters
        W_Q = np.random.randn(d_model, d_k)
        W_K = np.random.randn(d_model, d_k)
        W_V = np.random.randn(d_model, d_v)
        
        # Compute attention for this head
        head_output, _ = self_attention(x, W_Q, W_K, W_V)
        heads.append(head_output)
    
    # Concatenate all head outputs
    concat = np.concatenate(heads, axis=-1)  # Shape: (seq_len, n_heads * d_v)
    
    # Final linear projection to combine all heads
    W_O = np.random.randn(n_heads * d_v, d_model)
    output = concat @ W_O
    
    return output


def causal_attention(x, W_Q, W_K, W_V):
    """
    Causal (masked) attention for autoregressive language modeling.
    
    Prevents the model from attending to future tokens during training.
    This is essential for language models where each token should only
    depend on previous tokens, not future ones.
    
    Uses an upper triangular mask to set future positions to -infinity
    before the softmax, making their attention weights essentially zero.
    
    Args:
        x (np.ndarray): Input of shape (seq_len, d_model)
        W_Q, W_K, W_V (np.ndarray): Weight matrices for Q, K, V projections
        
    Returns:
        tuple: (output, attention_weights)
            - output: Masked attention output of shape (seq_len, d_v)
            - attention_weights: Causal attention scores of shape (seq_len, seq_len)
    """
    # Standard attention computation
    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V

    scores = Q @ K.T
    d_k = K.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)

    # Build causal mask: upper triangular matrix with 1s above diagonal
    seq_len = x.shape[0]
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)

    # Apply mask: set future positions to very negative value
    # After softmax, these become ~0, preventing attention to future tokens
    scaled_scores = scaled_scores.copy()
    scaled_scores[mask] = -1e9   # Could also use -np.inf

    attention_weights = softmax(scaled_scores)
    output = attention_weights @ V
    
    return output, attention_weights

def positional_encoding(seq_len, d_model):
    """
    Generate sinusoidal positional encodings for transformer models.
    
    Since transformers don't have inherent notion of position (unlike RNNs),
    we add positional information using sinusoidal functions of different
    frequencies. This allows the model to learn to attend by relative positions.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    The wavelengths form a geometric progression from 2π to 10000·2π.
    
    Args:
        seq_len (int): Maximum sequence length
        d_model (int): Model dimension (must be even)
        
    Returns:
        np.ndarray: Positional encoding matrix of shape (seq_len, d_model)
    """
    # Position indices: 0, 1, 2, ..., seq_len-1
    pos = np.arange(seq_len).reshape(-1, 1)
    
    # Dimension indices for sine: 0, 2, 4, ..., d_model-2
    i = np.arange(0, d_model, 2)
    
    # Compute the denominator: 10000^(2i/d_model)
    denom = np.power(10000, i / d_model)
    
    # Initialize positional encoding matrix
    pe = np.zeros((seq_len, d_model))
    
    # Apply sine to even dimensions and cosine to odd dimensions
    pe[:, 0::2] = np.sin(pos / denom)  # Even indices
    pe[:, 1::2] = np.cos(pos / denom)  # Odd indices
    
    return pe


def feed_forward(x, d_model, d_ff):
    """
    Position-wise Feed-Forward Network used in transformer blocks.
    
    Applies two linear transformations with ReLU activation in between:
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    This is applied to each position separately and identically. The inner
    layer has dimensionality d_ff, typically 4 times larger than d_model.
    
    Args:
        x (np.ndarray): Input of shape (seq_len, d_model)
        d_model (int): Model dimension
        d_ff (int): Feed-forward hidden dimension (usually 4 * d_model)
        
    Returns:
        np.ndarray: Output of shape (seq_len, d_model)
    """
    # Initialize random weights and zero biases
    # In practice, these would be learned parameters
    W1 = np.random.randn(d_model, d_ff)    # Expand to hidden size
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model)    # Project back to model size
    b2 = np.zeros(d_model)
    
    # First linear transformation + bias
    hidden = x @ W1 + b1
    
    # ReLU activation: max(0, x)
    hidden = np.maximum(0, hidden)
    
    # Second linear transformation + bias
    output = hidden @ W2 + b2
    
    return output


def layer_norm(x, epsilon=1e-6):
    """
    Layer Normalization for training stability and faster convergence.
    
    Normalizes inputs across the features (last dimension) for each example
    independently. Unlike batch normalization, this doesn't depend on batch
    statistics, making it suitable for variable-length sequences.
    
    LN(x) = γ * (x - μ) / (σ + ε) + β
    
    Here we assume γ=1 and β=0 for simplicity.
    
    Args:
        x (np.ndarray): Input tensor of any shape
        epsilon (float): Small value to prevent division by zero
        
    Returns:
        np.ndarray: Layer normalized output (same shape as input)
    """
    # Compute mean and standard deviation across the last dimension
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    
    # Normalize: (x - mean) / (std + epsilon)
    normalized = (x - mean) / (std + epsilon)
    
    return normalized


def transformer_block(x, n_heads, d_model, d_k, d_v, d_ff):
    """
    Complete Transformer Block combining all components.
    
    Implements the encoder layer from "Attention is All You Need":
    1. Multi-head self-attention with residual connection and layer norm
    2. Position-wise feed-forward with residual connection and layer norm
    
    The order here is: Sub-layer → Add & Norm
    
    Args:
        x (np.ndarray): Input of shape (seq_len, d_model)
        n_heads (int): Number of attention heads
        d_model (int): Model dimension
        d_k (int): Key/Query dimension per head
        d_v (int): Value dimension per head  
        d_ff (int): Feed-forward hidden dimension
        
    Returns:
        np.ndarray: Transformer block output of shape (seq_len, d_model)
    """
    # Sub-layer 1: Multi-head self-attention
    attn_output = multi_head_attention(x, n_heads, d_model, d_k, d_v)
    # Residual connection + layer normalization
    x = layer_norm(x + attn_output)
    
    # Sub-layer 2: Position-wise feed-forward network
    ff_output = feed_forward(x, d_model, d_ff)
    # Residual connection + layer normalization
    x = layer_norm(x + ff_output)
    
    return x