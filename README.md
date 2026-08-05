[README.md](https://github.com/user-attachments/files/30735890/README.md)
# CSI-Aware GNN Decoder for 5G NR LDPC Codes

> A Graph Neural Network decoder that explicitly injects real-time Channel State Information (CSI) into every message-passing iteration — achieving **2.0 dB coding gain** and **45% complexity reduction** over Weighted Belief Propagation (WBP) on 3GPP TS 38.212 compliant 5G NR codes.

---

## Problem

Standard LDPC decoders (Belief Propagation) fail under **Rayleigh flat-fading** channels because per-block LLR scaling errors accumulate with each iteration and cannot be recovered. Existing neural decoders like Weighted Belief Propagation (WBP) partially address this — but treat the wireless channel as a black box, ignoring instantaneous CSI that 5G NR receivers already have access to via pilot-based DMRS estimation.

**Result:** systematic BER degradation under fading that no amount of extra iterations can fix.

---

## Approach

This project proposes a **CSI-Aware GNN (CSI-GNN)** decoder that:

1. Operates on the bipartite Tanner graph of the 5G NR BG2-PCM1 LDPC code
2. Injects instantaneous noise variance **σ²** into every GNN message-passing iteration via a learned **tanh-log Dense projection layer**
3. Modulates soft message reliability proportionally to channel quality — directly compensating for per-block fading distortion
4. Maintains **fully parallelisable message-passing** (no recurrent bottlenecks unlike RNN-based approaches)

```
Input Codeword → BPSK Modulation → Rayleigh Fading Channel
     ↓
CSI-Aware LLR Computation  (σ² injected here)
     ↓
Tanner Graph Construction
     ↓
GNN Message Passing × T iterations  (σ² re-injected every iteration)
     ↓
Hard Decision → Decoded Codeword
```

---

## Results

All simulations validated on **3GPP TS 38.212 BG2-PCM1 LDPC codes** under Rayleigh flat-fading, using NVIDIA Sionna v0.14 + TensorFlow 2.12 on Google Colab T4 GPU.

| Metric | Value |
|--------|-------|
| **Coding gain over WBP** | Up to **2.0 dB** at Eb/N₀ = 3.0 dB |
| **BER improvement over WBP** | Up to **3.5×** (n=256, T=12) |
| **Complexity reduction** | **45%** average vs WBP |
| **Training time** | ~90 min on Colab T4 GPU |
| **Code lengths evaluated** | n = 128, 256, 384 |
| **JIT speedup** | 35% wall-clock reduction via `@tf.function` |

### BER vs Eb/N₀ (n=256, Rayleigh fading)

| Decoder | BER @ 3.0 dB (T=10) | BER @ 3.0 dB (T=12) |
|---------|---------------------|---------------------|
| WBP     | 3.21 × 10⁻⁴         | 2.60 × 10⁻⁴         |
| CSI-GNN | 2.14 × 10⁻⁴         | 1.33 × 10⁻⁴         |
| **Gain**| **1.5×**            | **2.0×**            |

---

## Architecture

### CSI Conditioning Mechanism

The key innovation is a **tanh-log projection layer** that transforms noise variance into an embedding-dimension bias vector, broadcast to all variable nodes at every iteration:

```python
CSI_bias = Dense(d, activation='tanh')(log(σ² + ε))
h_v(t) ← h_v(t) + CSI_bias          # injected every iteration
```

This ensures the decoder adapts to per-block channel quality in real time — not just at LLR initialisation.

### GNN Architecture

```
Variable Node Update (VNU):   MLP with ReLU activation, 3 layers, hidden dim 48
Check Node Update (CNU):      MLP with SiLU activation, 3 layers, hidden dim 48
Embedding dimension (d):      16
Decoder depths evaluated:     T = 2, 4, 6, 8, 10, 12, 15
Parameter count:              ~120K (T=10) · ~180K (T=15) · ~240K (T=20)
```

### Training Configuration

```
Loss:           Binary Cross-Entropy (BCE) on information bits
Optimiser:      Adam
LR schedule:    5×10⁻⁴ (35K steps) → 1×10⁻⁴ (200K steps) → 1×10⁻⁵ (200K steps)
SNR range:      Eb/N₀ ∈ [0, 6] dB (uniform sampling per mini-batch)
Batch size:     128
LLR clipping:   ±20
```

---

## Tech Stack

- **Python 3.10**
- **TensorFlow 2.12**
- **NVIDIA Sionna v0.14** — standards-compliant 5G NR simulation
- **Google Colab** (T4 GPU, 16 GB VRAM)
- **NumPy / Matplotlib** — analysis and plotting

---

## How to Run

```bash
git clone https://github.com/bsrikeesh/csi-aware-gnn-ldpc-decoder.git
cd csi-aware-gnn-ldpc-decoder
pip install -r requirements.txt
```

### Train the model

```bash
python train.py --code_length 256 --iterations 10 --snr_min 0 --snr_max 6
```

### Evaluate BER

```bash
python evaluate.py --code_length 256 --checkpoint checkpoints/gnn_t10.ckpt
```

### Run full BER sweep (all code lengths)

```bash
python sweep.py --lengths 128 256 384 --iterations 10 12 15
```

> **Note:** Training was done on Google Colab with a T4 GPU. A local GPU is recommended for full sweeps. CPU-only mode works but will be significantly slower.

---

## Code Structure

```
csi-aware-gnn-ldpc-decoder/
├── model/
│   ├── gnn_decoder.py       # CSI-GNN architecture
│   ├── csi_layer.py         # tanh-log projection layer
│   └── tanner_graph.py      # Tanner graph construction from H matrix
├── data/
│   └── bg2_pcm1.py          # 3GPP BG2-PCM1 parity-check matrix
├── train.py                 # Training loop
├── evaluate.py              # BER evaluation
├── sweep.py                 # Full BER sweep across lengths & depths
├── requirements.txt
└── README.md
```

---

## Paper

This project is based on the research paper:

**"Channel-Conditioned GNN Decoding of 5G NR LDPC Codes Under Rayleigh Flat Fading"**
B S Rikeesh, GITAM University, Bengaluru (2026)

Key contributions:
- First GNN decoder to explicitly inject σ² at every message-passing iteration via a tanh-log Dense projection layer
- End-to-end implementation on 3GPP TS 38.212 BG2-PCM1 LDPC codes using NVIDIA Sionna
- Ablation study confirming that direct noise variance injection is the source of performance gains

---

## Comparison with Prior Work

| Method | CSI Handling | Fading Adaptive | Parallelisable |
|--------|-------------|-----------------|----------------|
| Standard BP | LLR init only | ✗ | ✓ |
| WBP | LLR init only | ✗ | ✓ |
| GNN Decoder | Implicit (black box) | Partial | ✓ |
| RNN Decoder | Explicit (recursive) | ✓ | ✗ |
| **CSI-GNN (Ours)** | **Explicit (per-iteration)** | **✓** | **✓** |

---

## References

1. Cammerer et al., "Graph Neural Networks for Channel Decoding," IEEE JSAC, 2022
2. Nachmani et al., "Learning to Decode Linear Codes Using Deep Learning," ITW, 2016
3. Hoydis et al., "NVIDIA Sionna: An Open-Source Library for Next-Generation Physical Layer Research," arXiv:2203.11854
4. 3GPP TS 38.212, "NR; Multiplexing and Channel Coding," Release 17

---

## Author

**B S Rikeesh**
ECE Graduate · GITAM University, Bengaluru
AI/ML Engineer · GenAI Developer · ServiceNow CSA

[LinkedIn](https://linkedin.com/in/bsrikeesh) · [GitHub](https://github.com/bsrikeesh) · rikeeshbs@gmail.com
