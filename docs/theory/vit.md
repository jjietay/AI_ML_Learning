# Vision Transformer

## How it works (in actual ViT: patch_size=16*16)

### 1) Starting

- an image is split up to 8x8 patches (total patches = 64)
- each patches contains RGB values, so the standard 3 channels worth of pixel values (0 to 255)
- we normalize the values
- we then flatten each all 3 normalized channels of the patch and obtain the vector
- we do this for all the patches in the image and obtain a linear vector for each patch

### 2) Embeddings

- we transform them into embeddings vectors for each patch by taking each linear vector and passing it into the neural network to obtain an embedding vector for each patch
- these embedding vectors can be treated as word embeddings
- We also add an **Extra Learnable Embedding Vector** that we use to gather all important information from other patches using the 4th step: Attention block

### 3) Positional Embedding

- we add positional embedding to the Embedding vectors and then we take it to the attention block

### 4) Attention

- we take this embedding vector annd make 3 copies, and pass them into Query, Key, and Value matrices
- we get output Query, Key and Value vectors for the attention layer
- each embedding vector can be processed in parallel
-  we apply multiple attention blocks
- This attention block helps each patch to see its relationship with other patches
- Therefore, Attention mechanism in Vision Transformers gives it a **Global Receptive Field**


### 5) End

- After passing through many attention blocks
- we use a Neural Network with softmax for classification
- the input to the Neural Network is our extra learnable embedding vector created earlier during the embedding stage


## CNN vs ViT

- In CNN we have **Local Receptive Field**
- meaning it has a built-in inductive bias
- we can see textures because texture is a *local* feature
- CNN may predict class of image based on the texture in image
- On the otherhand, ViT has a **Global Receptive Field** that focusses more on *global features* like the shape of the object when classifying
