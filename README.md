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
├── attention.py       # Core transformer components
│   ├── softmax()
│   ├── self_attention()
│   ├── multi_head_attention()
│   ├── positional_encoding()
│   ├── feed_forward()
│   ├── layer_norm()
│   └── transformer_block()
│
└── generate.py        # Text generation
    ├── TransformerLM class
    ├── Vocabulary (11 words)
    └── generate_text()
```

---

## 🚀 Usage

**Test individual components:**
```python
from attention import self_attention, multi_head_attention, positional_encoding

# Self-attention
x = np.random.randn(5, 8)  # 5 words, 8-dim embeddings
output = self_attention(x, W_Q, W_K, W_V)

# Multi-head attention
output = multi_head_attention(x, n_heads=4, d_model=8, d_k=6, d_v=6)

# Positional encoding
pos_enc = positional_encoding(seq_len=5, d_model=8)
```

**Generate text:**
```bash
python3 generate.py
```

**Output (with random weights):**
```
<START> mat fast fast the ran mat on fast
<START> dog sat <PAD> on <START> cat the fast
<START> ran <PAD> cat <PAD> the sat cat cat
```

*Note: Gibberish because weights are random (not trained). But the architecture works!*

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

---

## 🏗️ Architecture
```
Input Text: "The cat sat"
        ↓
Tokenize: [3, 4, 5]
        ↓
Embeddings: (3, d_model)
        ↓
+ Positional Encoding
        ↓
┌─────────────────────────┐
│  Transformer Block 1    │
│  ├─ Multi-Head Attn     │
│  ├─ Add & Norm          │
│  ├─ Feed Forward        │
│  └─ Add & Norm          │
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
Output Projection
        ↓
Softmax (probabilities)
        ↓
Next Word: "on"
```

---

## 🎓 What I Learned

**Why Attention?**
- RNNs process sequentially → forget long-term context
- Attention sees ALL words at once → captures any relationship

**Why Multi-Head?**
- One head can only learn one pattern
- Multiple heads learn different relationships in parallel

**Why Positional Encoding?**
- Attention is mathematically position-blind
- "Dog bites man" vs "Man bites dog" look identical without position info

**Why Residual Connections?**
- Deep networks have vanishing gradient problem
- Skip connections let gradients flow directly through
- Allows stacking 100+ layers (GPT-3 has 96 blocks)

---

## 🔥 Key Insights

**This is the SAME architecture as:**
- GPT-3, GPT-4 (OpenAI)
- Llama 3.2 (Meta) 
- Claude (Anthropic)
- Mistral, Gemini, etc.

**Differences:**
- **Mine:** 11 words, 2 blocks, random weights
- **GPT-4:** 100K+ words, 96+ blocks, trained on trillions of tokens

**Same fundamental structure.** I now understand what happens inside these models.

---

## 📊 Components Breakdown

| File | Lines | What It Does |
|------|-------|--------------|
| `attention.py` | ~150 | All transformer building blocks |
| `generate.py` | ~80 | Text generation with simple vocabulary |

**Total:** Built a working transformer in ~230 lines of NumPy code.

---

## 🛠️ Technical Details

- **Language:** Python 3
- **Dependencies:** NumPy only
- **Model Size:** 
  - Vocabulary: 11 tokens
  - d_model: 16 (embedding dimension)
  - n_heads: 4 (attention heads)
  - n_blocks: 2 (transformer layers)
  - Parameters: ~3,500 (vs GPT-3's 175 billion)

---

## 🚧 What's NOT Implemented

- **Training loop** - Weights are random, not learned
- **Backpropagation** - No gradient computation
- **Optimization** - No Adam, no learning rate schedules
- **Real vocabulary** - Only 11 words, not 50K+ tokens
- **Scaling** - No GPU support, no batching optimizations

**These are engineering additions.** The core architecture is complete.

---

## 🎯 Next Steps

To make this a REAL language model:
1. Load real training data (Wikipedia, books, etc.)
2. Implement backpropagation for all layers
3. Add cross-entropy loss function
4. Train with Adam optimizer
5. Scale up (more blocks, larger vocabulary, GPU)

**Or:** Use this understanding to work with existing models (Hugging Face, fine-tuning, etc.)

---

## 📚 Resources

**Papers:**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [BERT](https://arxiv.org/abs/1810.04805) - Bidirectional transformers
- [GPT-3](https://arxiv.org/abs/2005.14165) - Language models at scale

**What I also built:**
- [ml-from-scratch](https://github.com/kush-3/ml-from-scratch) - Neural networks from scratch (97.58% MNIST)
- [paper-explainer](https://github.com/kush-3/paper-explainer) - AI-powered arXiv paper summarizer

---

## 📝 License

MIT

---

**Built by [Kush Patel](https://github.com/kush-3) to deeply understand transformers, not just use them.**

*"The best way to understand something is to build it from scratch."*