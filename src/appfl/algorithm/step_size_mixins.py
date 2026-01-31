from __future__ import annotations

"""Step-size / client-selection mixins - SASS VERSION

This implements the SASS (Stochastic Armijo Stochastic Search) algorithm
from the references in the PDF, NOT Algorithm 1.

Key insight from Miaolan: Each client does SASS iterations. The stepsize rule:
- If iteration is SUCCESSFUL (passes Armijo condition) → stepsize INCREASES
- If iteration is UNSUCCESSFUL (fails Armijo condition) → stepsize DECREASES

This is the opposite logic from Algorithm 1! Here:
- SELECTED = successful = INCREASE stepsize (be more aggressive)
- REJECTED = unsuccessful = DECREASE stepsize (be more conservative)

Reference: Jin, Scheinberg, Xie (2021) - "High probability complexity bounds 
for adaptive step search based on stochastic oracles"
"""

from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
import copy

import torch

StateDict = Dict[str, torch.Tensor]


def _flatten_grad_norm(grad_estimate: Dict[str, torch.Tensor]) -> float:
    """Compute the L2 norm of the gradient estimate (flattened)."""
    return torch.norm(torch.cat([g.view(-1) for g in grad_estimate.values()])).item()


class SGDMixin:
    """No selection: always use all clients (fixed step, classic FedAvg style)."""

    def select_clients_and_states(
        self, local_states: List[Any]
    ) -> Tuple[List[StateDict], List[int], Optional[Dict[int, float]]]:
        client_states: List[StateDict] = local_states
        selected = list(range(self.num_clients))
        return client_states, selected, None


class AdaptiveMixin:
    """SASS-based Adaptive FL Algorithm.
    
    Each client performs SASS iterations. Stepsize rule:
    - SUCCESSFUL iteration (passes Armijo) → INCREASE stepsize (multiply by γ)
    - UNSUCCESSFUL iteration (fails Armijo) → DECREASE stepsize (divide by γ)
    
    This is per-client adaptive stepsizes with Armijo-based acceptance.
    """

    def __init__(
        self,
        *args,
        server_lr: float = 0.01,      # Initial stepsize
        gamma: float = 1.2,            # Stepsize adjustment factor (γ > 1)
        max_lr: float = 1.0,           # Maximum allowed stepsize
        min_lr: float = 1e-8,          # Minimum allowed stepsize
        selection_rule: str = "per_client",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.server_lr = float(server_lr)
        self.gamma = float(gamma)
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        
        if self.gamma <= 1.0:
            raise ValueError(f"gamma must be > 1, got {self.gamma}")
        
        if selection_rule not in ("per_client", "sum_delta"):
            raise ValueError("selection_rule must be 'per_client' or 'sum_delta'")
        self.selection_rule = selection_rule

        # Initialize per-client stepsizes
        self.lr_clients: Dict[int, float] = {
            cid: self.server_lr for cid in range(self.num_clients)
        }
        
        self.round_num = 0

    def select_clients_and_states(
        self, local_states: List[Any]
    ) -> Tuple[List[StateDict], List[int], Dict[int, float]]:
        """
        SASS-based adaptive stepsizes.
        
        For each client:
        1. Check if Armijo condition is satisfied (sufficient decrease)
        2. If YES (successful) → INCREASE stepsize for next iteration
        3. If NO (unsuccessful) → DECREASE stepsize for next iteration
        4. Use only successful clients for global model update
        """
        self.round_num += 1
        
        print(f"\n{'='*80}")
        print(f"ROUND {self.round_num} - SASS Adaptive Stepsizes")
        print(f"{'='*80}")
        print(f"SASS Rule: Successful → α×{self.gamma}, Unsuccessful → α/{self.gamma}")
        
        if len(local_states) != self.num_clients:
            raise ValueError(f"Expected {self.num_clients} client states, got {len(local_states)}")

        device = self.device

        # --- Collect data from clients ---
        grad_estimates: List[StateDict] = [None] * self.num_clients  # type: ignore
        grad_norm_sq: Dict[int, float] = {}
        delta_f: Dict[int, float] = {}
        primal_states: List[StateDict] = [None] * self.num_clients  # type: ignore

        print(f"\nCurrent Stepsizes:")
        for cid in range(self.num_clients):
            print(f"  Client {cid}: α = {self.lr_clients[cid]:.6f}")

        for cid, state in enumerate(local_states):
            if not isinstance(state, dict) or "grad_estimate" not in state:
                raise TypeError(
                    "AdaptiveMixin expects dicts with keys: primal_state, grad_estimate, function_value_difference"
                )

            primal_states[cid] = {k: v.to(device) for k, v in state["primal_state"].items()}
            grad_est = {k: v.to(device) for k, v in state["grad_estimate"].items()}
            grad_estimates[cid] = grad_est
            gnorm = _flatten_grad_norm(grad_est)
            grad_norm_sq[cid] = gnorm * gnorm
            delta_f[cid] = float(state["function_value_difference"])

        # --- Armijo condition check (determines success/failure) ---
        print(f"\n{'='*80}")
        print(f"ARMIJO CONDITION CHECK (Sufficient Decrease)")
        print(f"{'='*80}")
        print(f"Condition: Δf̃ₖc ≤ -αₖc ||g̃ₖc||² (like Armijo with c=1)\n")
        print(f"{'Client':<8} {'Δf̃ₖc':<14} {'||g̃ₖc||²':<14} {'αₖc':<12} {'Threshold':<14} {'Result':<15}")
        print(f"{'-'*80}")

        successful_clients = []  # Clients that passed Armijo condition
        
        if self.selection_rule == "sum_delta":
            sum_delta = sum(delta_f.values())
            for cid in range(self.num_clients):
                threshold = -self.lr_clients[cid] * grad_norm_sq[cid]
                is_successful = sum_delta <= threshold
                if is_successful:
                    successful_clients.append(cid)
                result = "SUCCESS ✓" if is_successful else "FAIL ✗"
                print(f"{cid:<8} {delta_f[cid]:<14.6f} {grad_norm_sq[cid]:<14.6f} "
                      f"{self.lr_clients[cid]:<12.6f} {threshold:<14.6f} {result:<15}")
            print(f"\nRule: sum_delta, Sum: {sum_delta:.6f}")
        else:
            for cid in range(self.num_clients):
                threshold = -self.lr_clients[cid] * grad_norm_sq[cid]
                is_successful = delta_f[cid] <= threshold
                if is_successful:
                    successful_clients.append(cid)
                result = "SUCCESS ✓" if is_successful else "FAIL ✗"
                print(f"{cid:<8} {delta_f[cid]:<14.6f} {grad_norm_sq[cid]:<14.6f} "
                      f"{self.lr_clients[cid]:<12.6f} {threshold:<14.6f} {result:<15}")
            print(f"\nRule: per_client")

        print(f"\n{'='*80}")
        print(f"SUCCESSFUL CLIENTS: {successful_clients} ({len(successful_clients)}/{self.num_clients})")
        print(f"{'='*80}")

        # If no successful clients, use all (fallback)
        if len(successful_clients) == 0:
            print(f"\n⚠ WARNING: No successful clients, using all")
            successful_clients = list(range(self.num_clients))

        # --- Update global model (using only successful clients) ---
        print(f"\n{'='*80}")
        print(f"GLOBAL MODEL UPDATE")
        print(f"{'='*80}")
        print(f"Using {len(successful_clients)} successful clients")
        
        current_global_state = self.model.state_dict()
        new_global_state: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        total_step_norm = 0.0
        
        for name, param in self.model.named_parameters():
            if name in grad_estimates[successful_clients[0]]:
                avg_grad = torch.zeros_like(param, device=device)
                for cid in successful_clients:
                    avg_grad += self.lr_clients[cid] * grad_estimates[cid][name]
                avg_grad = avg_grad / float(len(successful_clients))
                total_step_norm += torch.sum(avg_grad ** 2).item()
                new_global_state[name] = current_global_state[name].to(device) - avg_grad
            else:
                new_global_state[name] = current_global_state[name].clone()
        
        print(f"Update step norm: {torch.sqrt(torch.tensor(total_step_norm)).item():.6f}")
        
        # For buffers
        for name, buffer in self.model.named_buffers():
            acc = torch.zeros_like(buffer, device=device)
            count = 0
            for cid in successful_clients:
                if name in primal_states[cid]:
                    acc += primal_states[cid][name]
                    count += 1
            if count > 0:
                new_global_state[name] = acc / float(count)
            else:
                new_global_state[name] = current_global_state[name].clone()

        # --- SASS stepsize update rule ---
        print(f"\n{'='*80}")
        print(f"SASS STEPSIZE UPDATE")
        print(f"{'='*80}")
        print(f"Rule: SUCCESS → α×{self.gamma} (INCREASE), FAIL → α/{self.gamma} (DECREASE)")
        print(f"\n{'Client':<8} {'Old α':<12} {'Result':<12} {'Operation':<20} {'New α':<12} {'Change':<10}")
        print(f"{'-'*90}")
        
        for cid in range(self.num_clients):
            old_lr = self.lr_clients[cid]
            
            if cid in successful_clients:
                # SUCCESS: INCREASE stepsize
                new_lr = old_lr * self.gamma
                operation = f"× {self.gamma}"
                result_str = "SUCCESS"
                change = "INCREASE ↑"
            else:
                # FAIL: DECREASE stepsize
                new_lr = old_lr / self.gamma
                operation = f"/ {self.gamma}"
                result_str = "FAIL"
                change = "DECREASE ↓"
            
            # Apply bounds
            new_lr = max(min(new_lr, self.max_lr), self.min_lr)
            
            self.lr_clients[cid] = new_lr
            
            print(f"{cid:<8} {old_lr:<12.6f} {result_str:<12} {operation:<20} "
                  f"{new_lr:<12.6f} {change:<10}")

        print(f"\nStepsize Summary:")
        print(f"  Min: {min(self.lr_clients.values()):.6f}")
        print(f"  Max: {max(self.lr_clients.values()):.6f}")
        print(f"  Mean: {sum(self.lr_clients.values())/len(self.lr_clients):.6f}")
        print(f"{'='*80}\n")

        return [new_global_state], successful_clients, self.lr_clients