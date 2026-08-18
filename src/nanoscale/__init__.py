"""NanoScale-LM: a small language model built from scratch, and made deployable.

The package is split into two arcs that mirror the project thesis:

*Arc 1: build the model* (:mod:`nanoscale.tokenizer`, :mod:`nanoscale.model`,
:mod:`nanoscale.optim`, :mod:`nanoscale.train`, :mod:`nanoscale.align`) implements a
modern decoder-only transformer (RoPE + RMSNorm + SwiGLU + GQA + QK-norm), the Muon
optimizer, a pretraining loop and the SFT/DPO/SimPO alignment stack.

*Arc 2: serve it cheaply* (:mod:`nanoscale.distill`, :mod:`nanoscale.quantize`,
:mod:`nanoscale.specdec`, :mod:`nanoscale.serve`) compresses that exact model with
knowledge distillation, post-training quantization (RTN / GPTQ / AWQ) and speculative
decoding.

Everything is implemented from first principles; no high-level trainer library is used
anywhere in this package.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
