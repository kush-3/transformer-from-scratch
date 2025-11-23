import numpy as np

def softmax(x):
    """Apply softmax along last dimension"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def self_attention(X, W_Q, W_K, W_V):
    """
    Scaled dot-product attention
    
    Args:
        X: Input (seq_len, d_model)
        W_Q, W_K, W_V: Weight matrices
        
    Returns:
        output: Attention output (seq_len, d_v)
        attention_weights: Attention scores (seq_len, seq_len)
    """
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    scores = Q @ K.T
    d_k = K.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)
    
    attention_weights = softmax(scaled_scores)
    output = attention_weights @ V
    
    return output, attention_weights


def multi_head_attention(x, n_heads, d_model, d_k, d_v):
    """Multi-head attention with random weights"""
    heads = []
    
    for i in range(n_heads):
        W_Q = np.random.randn(d_model, d_k)
        W_K = np.random.randn(d_model, d_k)
        W_V = np.random.randn(d_model, d_v)
        
        head_output, _ = self_attention(x, W_Q, W_K, W_V)
        heads.append(head_output)
    
    concat = np.concatenate(heads, axis=-1)
    W_O = np.random.randn(n_heads * d_v, d_model)
    output = concat @ W_O
    
    return output


def positional_encoding(seq_len, d_model):
    """
    Sinusoidal positional encoding
    
    Args:
        seq_len: Sequence length
        d_model: Model dimension
        
    Returns:
        Positional encoding matrix (seq_len, d_model)
    """
    pos = np.arange(seq_len).reshape(-1, 1)
    i = np.arange(0, d_model, 2)
    denom = np.power(10000, i / d_model)
    
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(pos / denom)
    pe[:, 1::2] = np.cos(pos / denom)
    
    return pe


def feed_forward(x, d_model, d_ff):
    """
    Feed-forward network: expand → ReLU → compress
    
    Args:
        x: Input (seq_len, d_model)
        d_model: Model dimension
        d_ff: Feed-forward hidden dimension
        
    Returns:
        Output (seq_len, d_model)
    """
    W1 = np.random.randn(d_model, d_ff)
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model)
    b2 = np.zeros(d_model)
    
    hidden = x @ W1 + b1
    hidden = np.maximum(0, hidden)  # ReLU
    output = hidden @ W2 + b2
    
    return output


def layer_norm(x, epsilon=1e-6):
    """Layer normalization"""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    normalized = (x - mean) / (std + epsilon)
    return normalized


def transformer_block(x, n_heads, d_model, d_k, d_v, d_ff):
    """
    Complete transformer block:
    - Multi-head attention + residual + norm
    - Feed-forward + residual + norm
    
    Args:
        x: Input (seq_len, d_model)
        n_heads: Number of attention heads
        d_model: Model dimension
        d_k, d_v: Key and value dimensions per head
        d_ff: Feed-forward hidden dimension
        
    Returns:
        Output (seq_len, d_model)
    """
    # Sub-layer 1: Multi-head attention
    attn_output = multi_head_attention(x, n_heads, d_model, d_k, d_v)
    x = layer_norm(x + attn_output)  # Residual + norm
    
    # Sub-layer 2: Feed-forward
    ff_output = feed_forward(x, d_model, d_ff)
    x = layer_norm(x + ff_output)  # Residual + norm
    
    return x