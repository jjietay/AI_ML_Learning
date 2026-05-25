# Self-Attention & Transformers

> *"Similar tokens cluster together in embedding space, and the directions of these vectors encode meaning."*

Self-attention is the mechanism that lets every token in a sequence directly attend to, and be influenced by every other token, regardless of distance (race, language or religion).

---

## Attention Mechanism

**Attention** just means *how much should word A influence the representation of another other random word B?*

- Attention weighs how much influence word A has on word B when the network is trying to understand the full context.

- Each pair of tokens gets an **attention weight** that scales (describes in numbers) how much one token's information flows (affects) into another's updated representation.

---

## Query, Key, and Value

The attention mechanism splits each token's embedding into three roles:

| Role | Analogy | What it encodes |
|---|---|---|
| **Query (Q)** | A question being asked | "What am I looking for?" |
| **Key (K)** | A label on a book | "What do I contain?" |
| **Value (V)** | The book's content | "What information do I carry?" |

Each is computed by multiplying the raw token embedding by a learned weight matrix:

```
Q = embedding @ W_Q     (smaller dimension — the "search space")
K = embedding @ W_K     (same smaller dimension)
V = embedding @ W_V     (full or projected dimension)
```

### Query

- Every word has a **query vector** that kinda “asks” surrounding tokens if they are related to one another
- The Query vectors is then calculated by multiplying the Query matrices into the raw embedding of each token
- Meaning each token would have an updated vector embedding that embeds Query vector into it, which can be perceived as a Query embedded vector
- The query vector lives in a smaller-dimensional space and its direction encodes 'what this token is looking for,' and it will be compared against the key vectors of other tokens.

### Key

- Conceptually, each tokens has a key vector which are the answers to the queries
- Key vectors are also calculated by multiplying the key matrices into the vector embeddings of each token, similar to the Query matrices
- The key embedded vectors are compared to the query embedded vectors in the smaller dimensional space and closely aligned vectors are matched together
- Closely aligned key–query pairs (high dot product) mean the two tokens are highly relevant to each other

### Key–Query Matching

- We dot product between each key-query pair
- The bigger magnitude of dots means they align better

Relevance between two tokens is measured by the **dot product** of their query and key vectors:

```
score(i, j) = Q_i · K_j
```

A larger dot product means better alignment. The full set of scores forms a grid called the **Attention Pattern**:

> *Proper Termininology: "The embedding of 'Blue' ATTENDS TO the embedding of 'creature'"* — meaning 'creature' is the query, 'Blue' is the key it matched with.

---

## Masking


- During training, we want training process to be more efficient so we simultaneously predict each possible word in the sentence instead of only the LAST one --> so predict subsequence of tokens too
- For example, "The happy dog jumps over the fence in ____"
- An obvious goal is to predict the last words
- But since we wait the training process to be more efficient, we can also feed "The happy dog jumps over the ____ in excitement" into the model for training (basically reusing the sequence)
- We need to prevent later tokens from influencing because it will kinda giveaway the answer for the earlier tokens
- How we do it is to set those specific dot values to negative infinite, and then we apply the softmax (talked about earlier) which forces these values to 0 (preventing them from influencing)

During training, the model simultaneously predicts every position in the sequence, so we need to **prevent later tokens from leaking information to earlier ones**:

1. Set the attention score for any future position to \(-\infty\)
2. After softmax, these become **0**, effectively blocking future tokens from influencing earlier ones

```
Scores (before mask):  [ 0.8,  0.5,   -0.2,   0.9]
Mask applied:          [ 0.8,  0.5,   -inf,  -inf]
After softmax:         [ 0.62, 0.38,   0.0,   0.0]
```

> **Context size note:** The attention pattern has size `context_length²`, so longer sequences are quadratically more expensive. Meaning big O notation is O(x²)

---

## Value - Updating the Embedding

Once attention weights are computed, the **Value vectors** are used to update each token's embedding:

1. Multiply each token's embedding by `W_V` --> get a value vector
2. Multiply each value vector by the corresponding attention weight
3. **Sum** the weighted value vectors for that position --> this gives a *delta* (Δ) representing what information flows in
4. Add Δ to the original embedding: `new_embedding = embedding + Δ`

---

## Cross-Attention

Cross-attention is used when a model processes **two different data streams** (e.g. text + image in a Vision-Language Model):

| Property | Self-Attention | Cross-Attention |
|---|---|---|
| Q source | Same sequence | Sequence A (e.g. decoder) |
| K, V source | Same sequence | Sequence B (e.g. encoder output) |
| Masking | Usually yes (causal) | Usually no |
| Use case | Understanding within one modality | Aligning across two modalities |

---

## Multi-Head Attention

Running a **single** attention head gives one view of token relationships. Running **multiple heads in parallel** lets different heads learn different relationship types simultaneously.

### How It Works

1. **Split** the embedding dimension across heads, e.g. 768-dim split into 12 heads × 64-dim each
2. Each head runs its own Q, K, V projections and attention computation independently
3. Each head still sees **all tokens**, just through a different 64-dim "lens"
4. **Concatenate** all head outputs back to 768-dim
5. Pass through a final projection matrix \(W_O\) to mix the heads together

```
head_1 (64-dim) ─┐
head_2 (64-dim) ─┤  concat (768-dim)  →  W_O  →  output (768-dim)
    ...          ─┤
head_n (64-dim) ─┘
```

Multiple transformer blocks (each with multi-head attention + MLP) are stacked on top of each other before the final prediction layer.
