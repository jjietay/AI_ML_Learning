# BERT

**Bidirectional Encoder Representations from Transformers (BERT)** is a language model that reads text in *both directions simultaneously* using the Transformer encoder. Unlike a standard language model that predicts the next word left-to-right, BERT can attend to every other word from both sides at once, giving it a much richer understanding of context.

## Key Difference from the Original Transformer

| | Transformer | BERT |
|---|---|---|
| Architecture used | Encoder + Decoder | **Encoder only** |
| Reading direction | Left-to-right (decoder is masked) | **Bidirectional** |
| Primary task | Sequence-to-sequence (translation) | Understanding / classification |
| Training objective | Next-token prediction | Masked Language Modelling + NSP |

---

## Pre-training Tasks

BERT learns through two self-supervised tasks on unlabelled text.

### 1. Masked Language Modelling (MLM)

BERT randomly masks 15% of tokens in each sentence and learns to predict the masked words from context:

- **80%** of selected tokens --> replaced with `[MASK]`
- **10%** of selected tokens --> replaced with a **random word**
- **10%** of selected tokens --> kept as the **original word**

```
Input:  "[CLS] how are [MASK] doing today [SEP]"
Target: predict "you" at the masked position
```

This forces the model to use bidirectional context rather than just left-to-right patterns.

### 2. Next Sentence Prediction (NSP)

BERT also learns whether two sentences naturally follow each other:

```
[CLS] Sentence A [SEP] Sentence B [SEP]
                                   ↓
                    Binary output: IsNext / NotNext
```

- The `[CLS]` token attends to all words in both sentences via self-attention
- Its final representation feeds into a binary classifier
- Achieves **~97% accuracy** on the NSP task

---

## Special Tokens

| Token | Purpose |
|---|---|
| `[CLS]` | Prepended to every input; its final embedding is used for classification tasks |
| `[SEP]` | Separates sentence A from sentence B; also marks end of input |
| `[MASK]` | Replaces tokens that the model must predict during MLM pre-training |

---

## Why It Works

- **Bidirectional attention**: every token sees every other token in both directions simultaneously, unlike GPT's left-to-right attention
- **Parallelisable**: self-attention replaces recurrence, so training is much faster than RNN-based models
- **Transfer learning**: pre-train once on large unlabelled text, then fine-tune on small labelled datasets for specific tasks (classification, QA, NER, etc.)
