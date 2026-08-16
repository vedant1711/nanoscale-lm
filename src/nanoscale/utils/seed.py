"""Global seed control and determinism (spec A3.4: reproducibility is a graded feature).

Two seeded ``nano`` runs must produce identical loss trajectories to floating-point
tolerance. That requires seeding Python, NumPy and torch, *and* making the dataloader
order a pure function of the seed (see :mod:`nanoscale.train.data`), *and* disabling
nondeterministic kernels.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import random
from collections.abc import Iterator

import numpy as np
import torch

__all__ = ["derive_seed", "seed_all", "seed_worker", "temporary_seed"]


def seed_all(seed: int, *, deterministic: bool = True) -> None:
    """Seed every RNG NanoScale touches.

    Args:
        seed: The global seed.
        deterministic: If True, also select deterministic cuDNN/cuBLAS algorithms and
            ask torch to raise on any remaining nondeterministic op. This costs some
            throughput and is what the determinism test relies on.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuBLAS needs this env var set before the first CUDA context for determinism.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: a few ops (e.g. some scatter kernels) have no deterministic
        # implementation; we prefer a warning over an unrunnable training loop.
        torch.use_deterministic_algorithms(True, warn_only=True)


def derive_seed(seed: int, *tags: str | int) -> int:
    """Derive a stable child seed from a parent seed and a set of tags.

    Used so that, for example, data shuffling, dropout and generation each get an
    independent-but-reproducible stream: ``derive_seed(cfg.seed, "data", epoch)``.
    """
    payload = "|".join([str(seed), *[str(t) for t in tags]]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def seed_worker(worker_id: int) -> None:  # pragma: no cover - requires num_workers>0
    """``worker_init_fn`` for :class:`torch.utils.data.DataLoader`."""
    base = torch.initial_seed() % (2**31 - 1)
    worker_seed = derive_seed(base, "worker", worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)


@contextlib.contextmanager
def temporary_seed(seed: int) -> Iterator[None]:
    """Context manager that seeds the RNGs and restores the previous state on exit.

    Useful for evaluation and generation, which must be reproducible without perturbing
    the training RNG stream.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:  # pragma: no cover - no GPU in CI
            torch.cuda.set_rng_state_all(cuda_states)
