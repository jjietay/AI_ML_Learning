# Self-Attention Head

main idea: `"classic intuition is similar tokens are closer together, and directions of these vectors encode meaning"`

## Attention Mechanism:

- Attention in transformers is just how much each word affects each other in the sentence or context.
- Attention weighs how much influence word A has on word B when the network is trying to understand the full context.

## Query:

- This has a smaller embedding than the raw vector embedding
- Every word has a query vector that kinda “asks” surrounding tokens if they are related to one another
- The Query vectors is then calculated by multiplying the Query matrices into the raw embedding of each token
- Meaning each token would have an updated vector embedding that embeds Query vector into it, which can be perceived as a Query embedded vector
- The query vector lives in a smaller-dimensional space; its direction encodes 'what this token is looking for,' and it will be compared against the key vectors of other tokens.

## Keys:

- Conceptually, each tokens has a key vector which are the answers to the queries
- Key vectors are also calculated by multiplying the key matrices into the vector embeddings of each token, similar to the Query matrices
- The key embedded vectors are compared to the query embedded vectors in the smaller dimensional space and closely aligned vectors are matched together

## Key-Query Pair:

- We dot product between each key-query pair
- The bigger magnitude of dots means they align better
- Proper term is: “Embedding of Blue, ATTEND TO the embedding of creature.”
- This means that the CREATURE’s embedding is the Query embedding and Blue’s embedding is the key embedding
- Each pair in table forms an Attention Pattern
- We then apply softmax to each column of the grid
- However, before that, we should do masking

## Masking:

- To prevent later tokens from influencing earlier ones (happens when we train model, because we want training process to be more efficient so we simultaneously predict each possible word in the sentence instead of only the LAST one —> so predict subsequence of tokens too)
- We need to prevent later tokens from influencing because it will kinda giveaway the answer for the earlier tokens
- How we do it is to set those specific dot values to negative infinite, and then we apply the softmax (talked about earlier) which forces these values to 0 (preventing them from influencing)

## Context Size:

- The attention pattern size is the squared of the context size

## Value:

- Multiply each word’s vector embedding with the value matrix to get a value vector
- Multiply this value vector with the corresponding dot weight in that column in the attention pattern
- Sum the column together with its rescaled values (embedded vector * value matrix) to get a change, delta E.
- We add this delta E to the original embedding 


# Cross Attention

- involved models that process 2 different types of data
- example text + image (VLM)
- only diff is key and query maps different datasets
- keys might come from 1 language, and query may come from another language
- attention pattern may show which word in 1 language correspond to which word in another
- typically no masking

# Multiple Heads

- run all self-attention head in parallel
- allows different heads to learn different things

## How it works:

- split the emedding dimensions to each head
- for example, head 1 sees all tokens through a 64-dim "lens"; head 2 sees the same tokens through a different 64-dim lens; head 3 through yet another, etc.
- each head still sees ALL the tokens
- different heads learn different patterns
- the head outputs are concatenated (stacked side-by-side back into a 768-dim vector) and run through one final projection matrix W_O
- it's a single matrix multiply that mixes them
- this whole multi-head attention step is just one piece of one transformer block
- many blocks stack on top of each other before the final next-word prediction at the very end
