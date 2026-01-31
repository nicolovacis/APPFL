from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch

from .server_fed_avg import ServerFedAvg
from .aggregation_mixins import GeometricMedianMixin
from .step_size_mixins import SGDMixin

StateDict = Dict[str, torch.Tensor]

class FedGMSGD(SGDMixin, GeometricMedianMixin, ServerFedAvg):
    """Geometric Median aggregation with all clients selected."""

    def update(self, local_states: List[Any]) -> Tuple[StateDict, Optional[Dict[int, float]]]:
        client_states, selected_clients, lr_clients = self.select_clients_and_states(local_states)
        aggregated = self.aggregate_models(client_states, selected_clients)

        self.global_state = aggregated
        self.model.load_state_dict(self.global_state)
        return self.global_state, lr_clients
