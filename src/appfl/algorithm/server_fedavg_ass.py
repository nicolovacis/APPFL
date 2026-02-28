from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch

from .server_fed_avg import ServerFedAvg
from .aggregation_mixins import AveragingMixin
from .step_size_mixins import AdaptiveMixin

StateDict = Dict[str, torch.Tensor]


class FedAvgASS(AdaptiveMixin, AveragingMixin, ServerFedAvg):
    """FedAvg with Adaptive Step Size Selection (SASS).

    SASS logic runs entirely on the clients.  The server simply
    aggregates the received client weights via weighted averaging
    (same pattern as ``FedAvgSGD``).
    """

    def update(self, local_states: List[Any]) -> Tuple[StateDict, Optional[Dict[int, float]]]:
        client_states, selected_clients, _ = self.select_clients_and_states(local_states)
        aggregated = self.aggregate_models(client_states, selected_clients)

        self.global_state = aggregated
        self.model.load_state_dict(self.global_state)
        return self.global_state, None
