# Gated Recurrent Units (GRU)

<div align="center">
  <img src="../images/gru_img1.png" alt="Alt text" width="600">
</div>

<div align="center">
  <img src="../images/gru_img2.png" alt="Alt text" width="600">
</div>

## Introduction
- 2 gates in GRU compared to 3 gates in LSTM

---
### Reset Gate, $r_t$
- similar to forget gate in LSTM
- takes previous hidden state ($h_{t-1}$) and current input ($x_t$) and multiply them with their corresponding weight matrices ($W_{xr}, W_{hr}$)
- add a bias ($b_r$) and apply the sigmoid activation function
- we then mutiply this ($r_t$) with the previous hidden state (multiply element wise)

---
### Candidate Hidden State, $\tilde{h}_{t}$
- it is a candidate for the next hidden state
- calculation for the next hidden state candidate heavily relies on the reset gate
- similar to cell state in LSTM
- use current input ($x_t$) and multiply it with weight matrix ($W_xh$)
- using previous hidden state ($h_{t-1}$) and multiplying it element wise with reset gate ($r_t$) previously calculated
- multiply this with a weight matrix ($W_{hh}$) and add a bias ($b_h$) and lastly applying a tanh function

---
### Update Gate, $h_t$
- selects what to transfer from the previous hidden state and what to select from the current candidate hidden state to the next step
- multiply current input ($x_t$) and previous hidden state ($h_{t-1}$) with respective weights ($W_{xz}$, $W_{hz}$) and then add bias ($b_z$)
- and then apply a sigmoid function, resulting in $z_t$
- to get the final hidden state output $h_t$: multiply $z_t$ element wise with candidate hidden state $\tilde{h}_{t}$
- multiply the mirroring of $z_t$ (1-$z_t$) element wise with previous hidden state  $h_{t-1}$
- add these 2 together, resulting in $h_t$
- so what we are doing is just getting what's important from the previous hidden state and what's important from candidate hidden state
- and transfering the final output to next step as the new hidden state output

---
### Comparison between LSTM and GRU:
- LSTM has 3 gates around 4 neural networks
- GRU has 2 gates around 3 neural networks
- LSTM has better long term memory due to more gates
- LSTM requires slightly more computation than GRU