import numpy as np
from attention import *

vocab = {
    "<PAD>": 0,
    "<START>": 1,
    "<END>": 2,
    "the": 3,
    "cat": 4,
    "sat": 5,
    "on": 6,
    "mat": 7,
    "dog": 8,
    "ran": 9,
    "fast": 10
}

idx_to_word = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size}")
print(f"Words: {list(vocab.keys())}")


class TransformerLM:  # Fixed: LM not Lm
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_blocks):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_blocks = n_blocks
        
        # Embedding matrix: vocab_size x d_model
        self.embedding = np.random.randn(vocab_size, d_model) * 0.1  # Fixed typo
        
        # Output projection: d_model x vocab_size
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.1
    
    def embed(self, token_ids):  # Fixed: snake_case
        """Convert token IDs to embeddings"""
        return self.embedding[token_ids]  # Fixed typo
    
    def forward(self, token_ids):  # Fixed: snake_case
        """
        Forward pass: predict next word probabilities
        
        Args:
            token_ids: List of token indices
            
        Returns:
            Probability distribution over vocabulary
        """
        seq_len = len(token_ids)
        
        # Step 1: Embed tokens
        x = self.embed(token_ids)
        
        # Step 2: Add positional encoding
        pos_enc = positional_encoding(seq_len, self.d_model)
        x = x + pos_enc
        
        # Step 3: Pass through transformer blocks
        for _ in range(self.n_blocks):
            x = transformer_block(
                x,
                n_heads=self.n_heads,
                d_model=self.d_model,
                d_k=self.d_model // self.n_heads,
                d_v=self.d_model // self.n_heads,
                d_ff=self.d_ff
            )
        
        # Step 4: Take last token (predicts next word)
        last_token = x[-1]
        
        # Step 5: Project to vocabulary
        logits = last_token @ self.output_proj
        
        # Step 6: Convert to probabilities
        probs = softmax(logits.reshape(1, -1)).flatten()
        
        return probs


def generate_text(model, start_token, max_length=10):
    """Generate text autoregressively"""
    tokens = [start_token]
    
    for _ in range(max_length):
        # Get next word probabilities
        probs = model.forward(tokens)
        
        # Sample next token
        next_token = np.random.choice(vocab_size, p=probs)
        
        # Stop if <END> generated
        if next_token == vocab["<END>"]:
            break
        
        tokens.append(next_token)
    
    # Convert to words
    words = [idx_to_word[t] for t in tokens]
    return " ".join(words)


# Create model
np.random.seed(42)
model = TransformerLM(
    vocab_size=vocab_size,
    d_model=16,
    n_heads=4,
    d_ff=64,
    n_blocks=2
)

# Generate text
print("\n" + "="*60)
print("GENERATING TEXT (random weights, not trained)")
print("="*60)

for i in range(3):
    text = generate_text(model, start_token=vocab["<START>"], max_length=8)
    print(f"{i+1}. {text}")