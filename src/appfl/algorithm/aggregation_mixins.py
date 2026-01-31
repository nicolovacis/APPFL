from __future__ import annotations

"""Aggregation mixins.

These mixins are designed to integrate with the existing APPFL server pipeline:
servers are instantiated as:

    server = eval(cfg.fed.servername)(weights, model, loss_fn, num_clients, device, **cfg.fed.args)

and are expected to call `self.model.load_state_dict(global_state)`.

We keep behavior consistent with the current codebase:
  - trainable parameters (named_parameters) use *client weights*
  - non-trainable buffers (BatchNorm running stats, etc.) use an unweighted mean
    over the selected clients (mirrors `server_federated.FedServer.update`).
"""

from collections import OrderedDict
from typing import Dict, List, Sequence

import torch

from .geometric_median import compute_geometric_median

StateDict = Dict[str, torch.Tensor]


def _weights_to_tensor(weights_dict: Dict[int, float], num_clients: int, device: torch.device) -> torch.Tensor:
    """Convert the repo's `weights` dict to a tensor aligned with client ids 0..N-1."""
    w = torch.zeros(num_clients, device=device, dtype=torch.float32)
    for i in range(num_clients):
        if i not in weights_dict:
            raise KeyError(f"weights is missing client id {i}")
        w[i] = float(weights_dict[i])
    s = w.sum()
    if s.item() <= 0:
        raise ValueError("Client weights must sum to a positive value")
    return w / s


def _renorm_selected(w: torch.Tensor, selected: Sequence[int]) -> torch.Tensor:
    """Renormalize weights for selected clients only."""
    sel = w[list(selected)]
    s = sel.sum()
    if s.item() <= 0:
        raise ValueError("Selected client weights sum to non-positive")
    return sel / s


class AveragingMixin:
    """FedAvg aggregation (weighted mean for parameters)."""

    def aggregate_models(self, client_states: List[StateDict], selected_clients: Sequence[int]) -> StateDict:
        if len(selected_clients) == 0:
            raise ValueError("selected_clients is empty")

        ref = client_states[selected_clients[0]]
        device = next(iter(ref.values())).device

        w_full = _weights_to_tensor(self.weights, self.num_clients, device=device)
        w_sel = _renorm_selected(w_full, selected_clients)

        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        param_names = set(self.list_named_parameters) if hasattr(self, "list_named_parameters") else set()

        for name in ref.keys():
            if name in param_names:
                # Trainable parameters: use weighted average
                acc = torch.zeros_like(ref[name], device=device)
                for j, cid in enumerate(selected_clients):
                    acc += w_sel[j] * client_states[cid][name].to(device)
                out[name] = acc
            else:
                # Buffers: unweighted average (consistent with FedServer)
                acc = torch.zeros_like(ref[name], device=device)
                for cid in selected_clients:
                    acc += client_states[cid][name].to(device)
                out[name] = acc / float(len(selected_clients))

        return out


class GeometricMedianMixin:
    """Geometric-median aggregation (robust to Byzantine attacks and outliers).
    
    Reference: Pillutla et al. (2022) "Robust Aggregation for Federated Learning"
    """

    def __init__(self, *args, gm_eps: float = 1e-6, gm_maxiter: int = 100, gm_ftol: float = 1e-10, **kwargs):
        super().__init__(*args, **kwargs)
        self.gm_eps = gm_eps
        self.gm_maxiter = gm_maxiter
        self.gm_ftol = gm_ftol

    def aggregate_models(self, client_states: List[StateDict], selected_clients: Sequence[int]) -> StateDict:
        if len(selected_clients) == 0:
            raise ValueError("selected_clients is empty")

        ref = client_states[selected_clients[0]]
        device = next(iter(ref.values())).device

        w_full = _weights_to_tensor(self.weights, self.num_clients, device=device)
        w_sel = _renorm_selected(w_full, selected_clients)

        # Select only the states from selected clients
        points = [client_states[cid] for cid in selected_clients]
        
        # Compute geometric median with client weights
        return compute_geometric_median(
            points,
            weights=w_sel,
            eps=self.gm_eps,
            maxiter=self.gm_maxiter,
            ftol=self.gm_ftol,
        )
