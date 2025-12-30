# Image Captioning / Image-Text Generation Model — From Scratch Implementation

> "I have not failed 10,000 times. I have not failed once. I have succeeded in proving that those 10,000 ways will not work. When I have eliminated the ways that will not work, I will find the way that will work." — Thomas Edison

This repository contains a from-scratch implementation of an image captioning / image-text generation system, built for research and educational purposes. The focus of this project is on architectural design, low-level implementation, and training infrastructure rather than on achieving state-of-the-art benchmark numbers.

The project aim was to implement a complete end-to-end pipeline for learning aligned representations between images and text and generating captions conditioned on visual input.

---

## Project Overview

The model follows an encoder–decoder style architecture:
- A vision encoder processes images into a sequence of visual tokens.
- A text decoder generates captions autoregressively conditioned on visual features.
- Cross-attention is used to align visual and textual representations.
- Several modern training and optimization techniques are incorporated to make training feasible on limited compute.

The entire system (attention, positional encoding, routing, decoding strategies, etc.) is implemented manually without relying on high-level model libraries.

---

## Model Architecture

### Vision Encoder
- ResNet-based convolutional backbone for feature extraction.
- Patch/token formation inspired by Vision Transformers (ViT-style tokenization).
- Self-attention layers over visual tokens.

### Text Decoder
- Autoregressive Transformer-style decoder.
- Causal masking for proper autoregressive behavior.
- Rotary Positional Embeddings (RoPE) for position encoding.
- Self-attention within the text stream and cross-attention over visual tokens.

### Cross-Modal Alignment
- Cross-attention layers between text tokens (queries) and image tokens (keys/values).
- Enables the model to ground language generation in visual content.

### Mixture of Experts (Optional)
- DeepSeek-style MoE feed-forward layers with:
  - Top-k routing
  - Auxiliary load-balancing loss
- Allows scaling capacity without linear compute growth.

---

## Training Setup

### Optimization
- Mixed precision training (FP16/BF16).
- Gradient checkpointing for memory efficiency.
- DeepSpeed ZeRO Stage-3 for optimizer and parameter sharding.

### Teacher Forcing
- Adaptive / scheduled teacher forcing to gradually reduce reliance on ground-truth tokens during training.

### Losses
- Autoregressive language modeling loss on text tokens.
- Contrastive image-text alignment loss (sigmoid-based CLIP-style formulation optimized for small batch sizes).
- Auxiliary routing loss for MoE layers (if enabled).

---

## Inference

The model supports multiple decoding strategies:
- Greedy decoding
- Beam search
- Top-p (nucleus) sampling

These allow exploration of quality-diversity tradeoffs during generation.

---

## Engineering & Infrastructure

- Implemented entirely in PyTorch with custom attention, masking, and routing logic.
- Integrated with Weights & Biases (W&B) for experiment tracking.
- Designed to run on constrained hardware (Kaggle GPUs, single A100 setups, etc.).

---

## Training Logs (Initial Attempt)

**Training Metrics from Early Experiments:**
```
Epoch: 1/3, Batch: 1792/1849, Global Step: 448
Step Loss: 0.5400, LR: 8.553289e-05
Aux_loss: 0.0478, Clip_loss: 0.0785, Ce_loss: 0.5312

Batch 152/157
Combined Loss: 0.6272, CE: 0.4395, AUX: 0.0477, CLIP: 0.1401
```

---

## Known Issues & Lessons Learned

This project represents an important learning journey. The initial implementation encountered several challenges:

1. **Model Output Quality**: The model produces gibberish text during inference, indicating issues with the training dynamics or architectural choices.

2. **MoE Routing Inefficiency**: The Mixture of Experts implementation uses nested for-loops (one over experts, one over sequence length), significantly hurting training throughput. A vectorized implementation would be substantially faster.

3. **Attention Masking**: The padding logic and attention masks are not correctly implemented, potentially causing the model to attend to padding tokens or violate causality constraints.

4. **Teacher Forcing**: The scheduled teacher forcing mechanism is not functioning as intended, which may prevent the model from learning proper autoregressive generation.

5. **Memory Management**: Memory usage grows exponentially during training, likely due to gradient accumulation issues, improper clearing of cached tensors, or activation checkpointing bugs.

6. **CLIP Loss Implementation**: Uses a sigmoid-based formulation (as recommended for small batch sizes in the paper), but the effectiveness needs validation.

---

## Future Directions

This project was made when I wasnt quite aware with deeper understanding of transformers, attention mechanisms, and training dynamics, the next iteration will address:

- Proper attention mask implementation with correct padding and causality handling
- Vectorized MoE routing without nested loops
- Fixed teacher forcing schedule with proper sampling
- Memory-efficient training with proper tensor lifecycle management
- Validation of cross-attention alignment between vision and language modalities
- Better debugging infrastructure for monitoring intermediate activations

This project embodies the spirit of iterative learning and resilience. Every bug fixed and every failure analyzed brings us closer to a working system.

---

## Acknowledgments

This project was built as a learning exercise to deeply understand multimodal deep learning from first principles. Special thanks to the open-source community for making research accessible.

---
