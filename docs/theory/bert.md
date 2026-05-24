# Bidirectional Encoder Representations from Transformers

## Introduction
- every word looks at every other word (self-attention which is also in transformers)
- but how this is different from transformers is that bert only has the encoder part of transformers, no decoder part
- self-attention makes it parallisable (training time falls, accuracy rises)
- bert comes after transformers

## How it works
- bert plays a guessing game
- each word in the sentence contains a [MASK] (unsupervised learning)
- for example, "[CLS] how are [MASK] doing today [SEP]"
- training data generator will first mask out 15% of the sentence
- 80% will recieve [MASK] token
- 10% will recieve a random word
- 10% will recieve the original word
- during fine-tuning bert will never see masked tokens in real texts
- this forces model to pay attention to context
- bert can look at text bi-directionally (more powerful)
- [CLS] and [SEP] is used for sentence seperation
- every input starts with the [CLS] token and the sentence A and then [SEP]/seperation token then [CLS] then sentence B
- [CLS] uses self-attention to look at every word in both sentences
- the final representation of [CLS] goes into binary classifier which checks if sentence B is "next" (after sentence A) or not
- achieving 97% accuracy

