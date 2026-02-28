from __future__ import annotations

"""Step-size / client-selection mixins — SASS VERSION

Client-side SASS logic (Armijo check, step-size adaptation) lives in
``ClientAdaptOptim``.  The server mixins here only handle client
selection / weight extraction before standard aggregation.

Utility helpers (Armijo condition, tentative SGD step, grad-norm,
deterministic-seed context manager) are also defined here so both client
and server can import from a single place.

Reference: Jin, Scheinberg, Xie (2021) — "High probability complexity
bounds for adaptive step search based on stochastic oracles"
"""

import contextlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

StateDict = Dict[str, torch.Tensor]


# ── SASS utility functions (ported from centralized Sass/utils.py) ──────────


def check_armijo_conditions_nls(
    step_size: float,
    loss: torch.Tensor,
    grad_norm: torch.Tensor,
    loss_next: torch.Tensor,
    theta: float,
    eps_f: float,
) -> bool:
    """Non-line-search Armijo condition (same semantics as centralized SASS).

    Returns ``True`` when the tentative step is *successful*:

        loss_next  <=  loss - step_size * theta * ||grad||^2 + 2 * eps_f
    """
    break_condition = loss_next - (loss - step_size * theta * grad_norm ** 2 + 2 * eps_f)
    return bool(break_condition <= 0)


def try_sgd_update(params, step_size, params_current, grad_current):
    """Apply a tentative SGD step: p_next = p_current - step_size * g."""
    for p_next, p_current, g_current in zip(params, params_current, grad_current):
        p_next.data = p_current - step_size * g_current


def compute_grad_norm(grad_list):
    """L2 norm of a list of gradient tensors."""
    grad_norm = 0.0
    for g in grad_list:
        if g is None:
            continue
        grad_norm += torch.sum(torch.mul(g, g))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm


def get_grad_list(params):
    return [p.grad for p in params]


@contextlib.contextmanager
def random_seed_torch(seed, device=0):
    """Fix torch/numpy RNG for deterministic forward passes (same as Sass/utils.py)."""
    cpu_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        gpu_rng_state = torch.cuda.get_rng_state(0)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(gpu_rng_state, device)


# ── Mixins ──────────────────────────────────────────────────────────────────


class SGDMixin:
    """No selection: always use all clients (fixed step, classic FedAvg style)."""

    def select_clients_and_states(
        self, local_states: List[Any]
    ) -> Tuple[List[StateDict], List[int], Optional[Dict[int, float]]]:
        client_states: List[StateDict] = local_states
        selected = list(range(self.num_clients))
        return client_states, selected, None


class AdaptiveMixin:
    """Server-side mixin for SASS-based adaptive FL.

    All SASS logic (Armijo check, tentative step, step-size update) runs
    **on the client** (see ``ClientAdaptOptim``).  The server receives
    plain model weights from each client and delegates to the aggregation
    mixin (FedAvg or Geometric Median).

    SASS hyper-parameters are accepted in ``__init__`` so that the shared
    ``cfg.fed.args`` dict does not cause unexpected-keyword errors, but
    they are **not used** server-side.
    """

    def __init__(
        self,
        *args,
        # Accept (and ignore) SASS hyper-parameters that arrive via cfg.fed.args
        server_lr: float = 0.01,
        gamma: float = 1.25,
        gamma_decr: float = 0.7,
        theta: float = 0.2,
        eps_f: float = 0.0,
        max_lr: float = 10.0,
        min_lr: float = 1e-8,
        selection_rule: str = "per_client",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Nothing stored — SASS state lives on the clients.

    def select_clients_and_states(
        self, local_states: List[Any]
    ) -> Tuple[List[StateDict], List[int], Optional[Dict[int, float]]]:
        """Extract client weight dicts and select all clients for aggregation."""
        client_states: List[StateDict] = local_states
        selected = list(range(self.num_clients))
        return client_states, selected, None
