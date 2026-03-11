# 🧠 Transformer From Scratch

A complete transformer architecture built from scratch using only NumPy - no PyTorch, no TensorFlow, just pure math and understanding.

Built to deeply understand how GPT, Llama, and Claude actually work under the hood.

---

## 🎯 What I Built

| Component | Implementation | Purpose |
|-----------|----------------|---------|
| **Self-Attention** | Q, K, V matrices + scaled dot-product | Core innovation - lets model see relationships between all words |
| **Multi-Head Attention** | 4+ parallel attention heads | Learn different types of relationships (subject-verb, adjective-noun, etc.) |
| **Positional Encoding** | Sine/cosine position vectors | Give model position information (attention is position-blind) |
| **Feed-Forward Network** | 2-layer dense network with ReLU | Process attention output into useful representations |
| **Layer Normalization** | Mean/std normalization | Stabilize training across layers |
| **Residual Connections** | Skip connections | Enable deep networks (100+ layers) |
| **Full Transformer Block** | All above combined | The building block of GPT/Llama |
| **Text Generation** | Embedding + sampling | Generate sequences word by word |

---

## 📁 Project Structure
```
transformer-from-scratch/
├── attention.py       # Core transformer components (heavily commented)
│   ├── softmax()                 # Numerically stable softmax activation
│   ├── self_attention()          # Scaled dot-product attention mechanism
│   ├── multi_head_attention()    # Parallel attention heads for diverse patterns
│   ├── causal_attention()        # Masked attention for autoregressive models
│   ├── positional_encoding()     # Sinusoidal position embeddings
│   ├── feed_forward()           # Position-wise MLP with ReLU
│   ├── layer_norm()             # Layer normalization for training stability
│   └── transformer_block()      # Complete encoder block with residuals
│
└── generate.py        # Transformer language model demo (heavily commented)
    ├── TransformerLM class      # Complete model with embeddings & generation
    ├── Vocabulary (11 words)    # Simple vocab for demonstration
    └── generate_text()          # Autoregressive text generation
```

---

## 🚀 Usage

**Test individual components:**
```python
from attention import self_attention, multi_head_attention, positional_encoding

# Self-attention with 5 words, 8-dimensional embeddings
x = np.random.randn(5, 8)  
W_Q = np.random.randn(8, 6)  # Query projection matrix
W_K = np.random.randn(8, 6)  # Key projection matrix  
W_V = np.random.randn(8, 6)  # Value projection matrix
output, attn_weights = self_attention(x, W_Q, W_K, W_V)

# Multi-head attention with 4 parallel heads
output = multi_head_attention(x, n_heads=4, d_model=8, d_k=6, d_v=6)

# Positional encoding for sequence length 5
pos_enc = positional_encoding(seq_len=5, d_model=8)
```

**Generate text:**
```bash
python3 generate.py
```

**Sample Output (with random weights):**
```
Vocabulary size: 11
Words: ['<PAD>', '<START>', '<END>', 'the', 'cat', 'sat', 'on', 'mat', 'dog', 'ran', 'fast']

============================================================
GENERATING TEXT (random weights, not trained)
============================================================
Note: Since weights are random, output will be nonsensical.
In practice, these weights would be learned from text data.
============================================================
1. <START> mat fast fast the ran mat on
2. <START> dog sat <PAD> on <START> cat the fast  
3. <START> ran <PAD> cat <PAD> the sat cat cat

============================================================
ARCHITECTURE SUMMARY
============================================================
Model parameters:
  - Vocabulary size: 11
  - Embedding dimension: 16
  - Attention heads: 4
  - Feed-forward size: 64
  - Transformer blocks: 2
  - Total embedding params: 176
  - Total output params: 176
============================================================
```

*Note: Output is gibberish because weights are random (not trained). But the architecture works!*

---

## 🧮 The Math

**Self-Attention Formula:**
```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

**Where:**
- Q (Query): "What am I looking for?"
- K (Key): "What do I contain?"  
- V (Value): "What information do I provide?"
- √d_k: Scaling factor to prevent vanishing gradients

**Multi-Head Attention:**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O

where head_i = Attention(Q×W_Q_i, K×W_K_i, V×W_V_i)
```

**Positional Encoding:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Layer Normalization:**
```
LayerNorm(x) = (x - μ) / (σ + ε)
where μ = mean(x), σ = std(x)
```

---

## 🏗️ Architecture Flow
```
Input Text: "The cat sat"
        ↓
Tokenize: [3, 4, 5] (vocab lookup)
        ↓
Token Embeddings: (3, d_model) - learned dense vectors
        ↓
+ Positional Encoding (sine/cosine patterns)
        ↓
┌─────────────────────────┐
│  Transformer Block 1    │
│  ├─ Multi-Head Attn     │ ← Parallel attention heads
│  ├─ Add & Norm          │ ← Residual + layer norm
│  ├─ Feed Forward        │ ← 2-layer MLP with ReLU  
│  └─ Add & Norm          │ ← Residual + layer norm
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│  Transformer Block 2    │
│  ├─ Multi-Head Attn     │
│  ├─ Add & Norm          │
│  ├─ Feed Forward        │
│  └─ Add & Norm          │
└─────────────────────────┘
        ↓
Take Last Token Hidden State
        ↓
Output Projection: (d_model → vocab_size)
        ↓
Softmax → Probability Distribution
        ↓
Sample Next Word: "on" (token 6)
```

---

## 🎓 What I Learned

**Why Attention Works:**
- **Problem:** RNNs process sequentially → information bottleneck, forget long context
- **Solution:** Attention sees ALL positions simultaneously → any word can attend to any other
- **Result:** Parallel processing + unlimited context window (within sequence length)

**Why Multi-Head Attention:**
- Single attention head learns one type of relationship
- Multiple heads learn diverse patterns in parallel:
  - Head 1: Subject-verb relationships  
  - Head 2: Adjective-noun relationships
  - Head 3: Long-range dependencies
  - Head 4: Syntactic structure
- Concatenate all heads for rich representations

**Why Positional Encoding:**
- Attention mechanism is **position-invariant** by design
- "Dog bites man" and "Man bites dog" would be identical without position info
- Sinusoidal encoding adds position information that the model can learn to use
- Different frequencies allow the model to learn both local and global position patterns

**Why Residual Connections:**
- **Vanishing gradient problem:** Deep networks lose gradients in backpropagation
- Skip connections provide direct gradient flow from output to input
- Enables training very deep networks (GPT-3 has 96 layers!)
- Also helps with training stability and convergence speed

**Why Layer Normalization:**
- Normalizes activations within each layer to have mean=0, std=1
- Prevents internal covariate shift during training
- More stable than batch normalization for variable-length sequences
- Faster convergence and better gradient flow

---

## 🔥 Key Insights

**This is the EXACT same architecture as:**
- **GPT-3, GPT-4** (OpenAI) - Decoder-only transformers
- **Llama 3.2** (Meta) - Same architecture, different training
- **Claude** (Anthropic) - Transformer-based with safety training
- **Mistral, Gemini, etc.** - All use transformer blocks

**Scale Differences:**
- **My Implementation:** 11 tokens, 2 blocks, ~3,500 parameters, random weights
- **GPT-4:** 100K+ tokens, 96+ blocks, ~1.7 trillion parameters, trained on internet-scale data
- **Training:** Mine uses random weights; real models train for months on massive compute

**Same fundamental math and architecture.** I now understand what happens inside these billion-parameter models!

---

## 📊 Components Breakdown

| File | Lines | Purpose | Key Functions |
|------|-------|---------|---------------|
| `attention.py` | ~300 | Core transformer math | All attention mechanisms, positional encoding, transformer blocks |
| `generate.py` | ~200 | Language model demo | TransformerLM class, text generation, vocabulary handling |
| `README.md` | ~300 | Documentation | Architecture explanation, usage examples, mathematical formulas |

**Total:** Built a working transformer in ~500 lines of heavily commented NumPy code.

---

## 🛠️ Technical Implementation Details

**Dependencies:**
- **NumPy only** - No deep learning frameworks
- Pure mathematical implementation for educational clarity

**Model Architecture:**
- **Vocabulary:** 11 tokens (including special tokens)
- **d_model:** 16 (embedding/hidden dimension)
- **n_heads:** 4 (attention heads)  
- **d_ff:** 64 (feed-forward hidden size, typically 4×d_model)
- **n_blocks:** 2 (transformer layers)
- **Parameters:** ~3,500 total (vs GPT-3's 175 billion)

**Key Design Choices:**
- **Decoder-only architecture** (like GPT) for autoregressive generation
- **Causal masking** to prevent attention to future tokens
- **Random weight initialization** (not trained) to demonstrate architecture
- **Simple vocabulary** for clear demonstration of concepts

---

## 🚧 What's NOT Implemented (Yet)

**Training Infrastructure:**
- ❌ **Backpropagation** - No gradient computation through the network
- ❌ **Loss functions** - No cross-entropy loss for training
- ❌ **Optimizers** - No Adam, SGD, or learning rate schedules
- ❌ **Training loop** - No data loading, batching, or parameter updates

**Production Features:**
- ❌ **GPU acceleration** - CPU-only NumPy implementation
- ❌ **Batch processing** - Processes one sequence at a time
- ❌ **Real tokenization** - No BPE, no subword handling
- ❌ **Model checkpointing** - No saving/loading trained weights
- ❌ **Attention visualization** - No tools to inspect learned patterns

**These are engineering additions.** The core mathematical architecture is complete and correct.

---

## 🎯 Next Steps to Make This Production-Ready

**1. Add Training Infrastructure:**
```python
# Implement backpropagation for all layers
def backward_pass(model, loss, learning_rate):
    # Compute gradients via chain rule
    # Update all parameters (embeddings, attention weights, etc.)
    pass

# Add cross-entropy loss for language modeling  
def compute_loss(predicted_probs, target_tokens):
    return -np.log(predicted_probs[target_tokens]).mean()
```

**2. Scale Up the Model:**
- Increase vocabulary to 50K+ tokens (BPE tokenization)
- Add more transformer blocks (6-96 layers)  
- Increase d_model to 512-4096 dimensions
- Add more attention heads (8-32 heads)

**3. Add Real Training Data:**
- Load text datasets (Wikipedia, books, web crawl)
- Implement data preprocessing and batching
- Add sequence packing for efficiency

**4. Production Optimizations:**
- Port to PyTorch/JAX for GPU acceleration
- Implement gradient checkpointing for memory efficiency  
- Add mixed precision training (FP16)
- Implement model parallelism for large scales

**Or:** Use this understanding to work effectively with existing models (Hugging Face transformers, fine-tuning, prompt engineering, etc.)

---

## 📚 Educational Resources

**Foundation Papers:**
- 📄 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper (Vaswani et al., 2017)
- 📄 [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) - GPT-2 paper
- 📄 [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - GPT-3 paper

**Implementation Guides:**
- 🔗 [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanations
- 🔗 [Transformer from Scratch](https://peterbloem.nl/blog/transformers) - Mathematical walkthrough  
- 🔗 [GPT from Scratch](https://jaykmody.com/blog/gpt-from-scratch/) - Code implementation guide

**My Related Projects:**
- 🧠 [ml-from-scratch](https://github.com/kush-3/ml-from-scratch) - Neural networks, CNNs from scratch (97.58% MNIST)
- 📖 [paper-explainer](https://github.com/kush-3/paper-explainer) - AI-powered arXiv paper summarizer using transformers

---

## 🤝 Contributing

This is an educational project, but contributions are welcome:

**Code Improvements:**
- Add training infrastructure (backpropagation, optimizers)
- Implement attention visualization tools
- Add more efficient batching and GPU support
- Create unit tests for all components

**Documentation:**
- Add more mathematical derivations
- Create interactive Jupyter notebooks  
- Add architecture diagrams and visualizations
- Improve code comments and docstrings

**Extensions:**
- Implement BERT-style bidirectional attention
- Add different positional encoding schemes (learned, rotary)
- Implement other transformer variants (Switch Transformer, etc.)

---

## 📜 License

MIT License - Feel free to use this code for learning, teaching, or building upon.

---

## 👨‍💻 Author

**Built by [Kush Patel](https://github.com/kush-3)**

*"The best way to understand something is to build it from scratch."*

This project represents my journey to deeply understand transformers - not just use them as black boxes, but comprehend the mathematical foundations that power modern AI systems like GPT, Claude, and Llama.

**Connect with me:**
- 🐙 GitHub: [@kush-3](https://github.com/kush-3)
- 💼 LinkedIn: [Kush Patel](https://linkedin.com/in/kush-patel-ai)
- 📧 Email: kush.ai.research@gmail.com

---

**⭐ Star this repo if it helped you understand transformers better!**