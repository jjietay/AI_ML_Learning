# Recurrent Neural Network (RNN)

## 1. The Unrolled Network & The "Hidden State"
- **The Core Concept:** Unlike standard Feedforward Neural Networks where inputs are processed independently, an RNN contains a loop that allows information to be passed from one step to the next.
- **Unrolling:** When "unrolled" over time, an RNN looks like a sequence of standard neural networks, one for each timestep. 
- **The Hidden State:** The network takes two inputs at any given timestep: the current data point and the **hidden state** (the output/memory from the previous timestep).
- **Weight Sharing:** Crucially, the exact same weights and biases are used across every single timestep in the unrolled network. This is what allows the model to process sequences of varying lengths.

## 2. Main Formulas
The operations inside a standard RNN cell are typically broken down into two steps: updating the memory and calculating the prediction.

**1. The Hidden State Update (Memory):**
$$h_t = \alpha (W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
- **$h_t$**: The current hidden state (memory) for timestep $t$.
- **$h_{t-1}$**: The hidden state from the previous timestep $t-1$.
- **$x_t$**: The current input data at timestep $t$.
- **$W_{xh}$**: Weights applied to the current input.
- **$W_{hh}$**: Weights applied to the previous hidden state.
- **$b_h$**: The bias for the hidden state.
- **$\alpha$**: The activation function (typically $\tanh$ or $\text{ReLU}$, which introduces non-linearity).

**2. The Output Calculation (Prediction):**
$$y_t = W_{hy} h_t + b_y$$
- **$y_t$**: The output at timestep $t$.
- **$W_{hy}$**: Weights applied to the current hidden state to generate the output.
- **$b_y$**: The bias for the output.

## 3. Types of RNNs

1. **Many-to-Many (Synced) / Sequence-to-Sequence**
   - For every input given at a timestep, an output is generated for that specific timestep.
   - *Example:* Stock price forecasting over time, or classifying every frame in a video.

2. **Many-to-One / Sequence-to-Vector**
   - The network processes the entire sequence, but only outputs a single value at the very end. The outputs at intermediate timesteps are ignored.
   - *Example:* Sentiment analysis (reading a whole sentence to output "positive" or "negative") or email spam detection (yes/no).

3. **One-to-Many / Vector-to-Sequence**
   - We give the network a single input at the first timestep, and it outputs a sequence of data over multiple timesteps.
   - *Example:* Image captioning (input is one image vector, output is a sequence of words describing it).

4. **Many-to-Many (Delayed) / Encoder-Decoder**
   - The first part of the model (Encoder) processes the input sequence over several timesteps and compresses it into a single context vector. The second part (Decoder) then uses that vector to generate an output sequence.
   - *Example:* Machine translation (reading an entire sentence in English before outputting the translation in French).

## 4. Backpropagation Through Time (BPTT)
To train an RNN, we use a specialized version of standard backpropagation designed for sequential data.

### Standard Backpropagation vs. BPTT
In a normal neural network, backpropagation calculates the error (loss) at the output and passes the gradients backward through the network layers to update the weights. In an RNN, because the network loops over time, we must **unroll** the network and pass the gradients back through the timesteps.

### The BPTT Process
1. **Forward Pass:** The unrolled network calculates outputs from $t=1$ to $t=T$ (the end of the sequence).
2. **Calculate Loss:** A cost function evaluates the error of the predictions at the relevant timesteps.
3. **Backward Pass:** Gradients are calculated starting from the final timestep $T$ and moving backward to $t=1$.
4. **Summing the Gradients:** Because the RNN uses the *same* weight matrices ($W_{xh}$, $W_{hh}$, $W_{hy}$) at every timestep, the total gradient for a specific weight is the **sum** of the gradients calculated at each individual timestep.
5. **Weight Update:** The weights are updated using an optimizer (like SGD or Adam) to minimize the loss.

### The Fatal Flaw: Vanishing & Exploding Gradients
BPTT relies heavily on the Chain Rule of calculus. When passing gradients back from timestep 100 to timestep 1, the network must multiply the weight matrix $W_{hh}$ by itself 100 times.
- **Vanishing Gradient:** If the values in $W_{hh}$ are less than $1$, multiplying them repeatedly causes the gradients to shrink exponentially towards zero. When the gradient becomes zero, the network stops adjusting its weights. As a result, the RNN "forgets" early inputs and fails to learn long-term dependencies.
- **Exploding Gradient:** Conversely, if the weights are greater than $1$, the gradients multiply to massive numbers, causing the model calculations to overflow and crash (often resulting in `NaN` losses).

## 5. How it Works: The Matrix / Function View
To visualize how memory compresses over time, assume the function of a single RNN cell is $f$ and our input sequence is $[x_1, x_2, x_3]$.
- **$t=1$:** $h_1 = f(x_1, h_0)$ *(where $h_0$ is usually initialized as an array of zeros)*
- **$t=2$:** $h_2 = f(x_2, h_1) \rightarrow f(x_2, f(x_1, h_0))$
- **$t=3$:** $h_3 = f(x_3, h_2) \rightarrow f(x_3, f(x_2, f(x_1, h_0)))$

By timestep 3, the hidden state $h_3$ contains deeply nested, mathematically compressed information from the very first input $x_1$.