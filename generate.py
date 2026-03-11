"""
Transformer Language Model - Text Generation Demo

This module demonstrates a simple transformer-based language model for text generation.
It includes:
- A small vocabulary with basic English words
- TransformerLM class that combines embeddings, positional encoding, and transformer blocks
- Autoregressive text generation using sampling from predicted probability distributions

Note: This uses random weights (not trained) to show the architecture in action.
In practice, these weights would be learned through backpropagation on text data.
"""

import numpy as np
from attention import *

# Simple vocabulary for demonstration
# In practice, this would be much larger (thousands to millions of tokens)
vocab = {
    "<PAD>": 0,    # Padding token for batch processing
    "<START>": 1,  # Beginning of sequence token
    "<END>": 2,    # End of sequence token
    "the": 3,      # Common English words for simple sentences
    "cat": 4,
    "sat": 5,
    "on": 6,
    "mat": 7,
    "dog": 8,
    "ran": 9,
    "fast": 10
}

# Reverse mapping: token ID → word string
idx_to_word = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size}")
print(f"Words: {list(vocab.keys())}")


class TransformerLM:
    """
    Transformer Language Model for autoregressive text generation.
    
    Architecture:
    1. Token Embedding: Maps discrete tokens to continuous vectors
    2. Positional Encoding: Adds position information to embeddings  
    3. Transformer Blocks: Stack of multi-head attention + feed-forward layers
    4. Output Projection: Maps final hidden states to vocabulary probabilities
    
    This is a decoder-only architecture suitable for language modeling tasks.
    """
    
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_blocks):
        """
        Initialize the transformer language model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            d_model (int): Dimension of model embeddings and hidden states
            n_heads (int): Number of attention heads in each layer
            d_ff (int): Hidden dimension of feed-forward networks
            n_blocks (int): Number of transformer blocks to stack
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_blocks = n_blocks
        
        # Token embedding matrix: maps token IDs to dense vectors
        # Shape: (vocab_size, d_model)
        # Each row represents the embedding for one token in the vocabulary
        self.embedding = np.random.randn(vocab_size, d_model) * 0.1
        
        # Output projection matrix: maps hidden states back to vocabulary logits
        # Shape: (d_model, vocab_size) 
        # Used to predict probability distribution over next tokens
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.1
    
    def embed(self, token_ids):
        """
        Convert token IDs to their corresponding embedding vectors.
        
        Args:
            token_ids (list or np.ndarray): List of token indices
            
        Returns:
            np.ndarray: Embeddings of shape (len(token_ids), d_model)
        """
        return self.embedding[token_ids]
    
    def forward(self, token_ids):
        """
        Forward pass: predict next word probabilities given input sequence.
        
        This implements the core language modeling objective: given a sequence
        of tokens, predict the probability distribution over the next token.
        
        Args:
            token_ids (list): List of token indices representing input sequence
            
        Returns:
            np.ndarray: Probability distribution over vocabulary for next token
                       Shape: (vocab_size,) with values summing to 1.0
        """
        seq_len = len(token_ids)
        
        # Step 1: Convert token IDs to dense embeddings
        # This gives us a continuous representation we can do math with
        x = self.embed(token_ids)  # Shape: (seq_len, d_model)
        
        # Step 2: Add positional encoding
        # Transformers have no inherent position awareness, so we add
        # sinusoidal patterns to encode position information
        pos_enc = positional_encoding(seq_len, self.d_model)
        x = x + pos_enc  # Element-wise addition
        
        # Step 3: Pass through stack of transformer blocks
        # Each block applies self-attention and feed-forward processing
        for block_idx in range(self.n_blocks):
            x = transformer_block(
                x,
                n_heads=self.n_heads,
                d_model=self.d_model,
                d_k=self.d_model // self.n_heads,  # Split model dim across heads
                d_v=self.d_model // self.n_heads,
                d_ff=self.d_ff
            )
        
        # Step 4: Take the last token's hidden state
        # For language modeling, we only care about predicting the NEXT token
        # The last position's representation contains info about the full sequence
        last_token = x[-1]  # Shape: (d_model,)
        
        # Step 5: Project to vocabulary space
        # Convert hidden representation to logits (unnormalized scores) for each token
        logits = last_token @ self.output_proj  # Shape: (vocab_size,)
        
        # Step 6: Convert logits to probabilities using softmax
        # This gives us a proper probability distribution we can sample from
        probs = softmax(logits.reshape(1, -1)).flatten()
        
        return probs


def generate_text(model, start_token, max_length=10):
    """
    Generate text autoregressively using the trained language model.
    
    Autoregressive generation means we predict one token at a time, then
    feed that prediction back as input to predict the next token, and so on.
    This is how GPT-style models generate coherent sequences.
    
    Args:
        model (TransformerLM): The language model to use for generation
        start_token (int): Token ID to start generation with (usually <START>)
        max_length (int): Maximum number of tokens to generate
        
    Returns:
        str: Generated text as a space-separated string
    """
    # Initialize with the start token
    tokens = [start_token]
    
    # Generate tokens one by one
    for step in range(max_length):
        # Get probability distribution over next token
        probs = model.forward(tokens)
        
        # Sample next token from the probability distribution
        # This introduces randomness - could also use argmax for greedy decoding
        next_token = np.random.choice(vocab_size, p=probs)
        
        # Stop generation if we sample the <END> token
        if next_token == vocab["<END>"]:
            break
        
        # Add the sampled token to our sequence
        tokens.append(next_token)
    
    # Convert token IDs back to words for human readability
    words = [idx_to_word[t] for t in tokens]
    return " ".join(words)


# ============================================================================
#                           DEMO: TEXT GENERATION
# ============================================================================

# Create a transformer language model with random weights
# Set random seed for reproducible results
np.random.seed(42)

model = TransformerLM(
    vocab_size=vocab_size,    # 11 tokens in our small vocabulary
    d_model=16,              # Small embedding dimension for demo
    n_heads=4,               # 4 attention heads
    d_ff=64,                 # Feed-forward hidden size (4x d_model)
    n_blocks=2               # 2 transformer layers
)

# Generate some example text
print("\n" + "="*60)
print("GENERATING TEXT (random weights, not trained)")
print("="*60)
print("Note: Since weights are random, output will be nonsensical.")
print("In practice, these weights would be learned from text data.")
print("="*60)

# Generate multiple samples to show the randomness
for i in range(3):
    text = generate_text(model, start_token=vocab["<START>"], max_length=8)
    print(f"{i+1}. {text}")

print("\n" + "="*60)
print("ARCHITECTURE SUMMARY")
print("="*60)
print(f"Model parameters:")
print(f"  - Vocabulary size: {vocab_size}")
print(f"  - Embedding dimension: {model.d_model}")
print(f"  - Attention heads: {model.n_heads}")
print(f"  - Feed-forward size: {model.d_ff}")
print(f"  - Transformer blocks: {model.n_blocks}")
print(f"  - Total embedding params: {model.embedding.size:,}")
print(f"  - Total output params: {model.output_proj.size:,}")
print("="*60)