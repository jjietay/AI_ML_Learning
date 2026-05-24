## Vanishing Gradient Problem

- this happens when the network is very deep
- during backpropagation, loss.backward() calculates gradients and sends them backwards through every layer
- in a deep network, when the gradient reaches the earlier hidden layers
- it has been multiplied so many times it becomes super tiny
- leading to the term vanishing gradient
- when gradient vanishes:
    1) Early layers stop learning
    2) Only the last few layers update
    3) Deep networks perform worse than shallow ones