from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch

from .server_fed_avg import ServerFedAvg
from .aggregation_mixins import AveragingMixin
from .step_size_mixins import SGDMixin

StateDict = Dict[str, torch.Tensor]

class FedAvgSGD(SGDMixin, AveragingMixin, ServerFedAvg):
    """FedAvg with all clients selected (standard approach)."""

    def update(self, local_states: List[Any]) -> Tuple[StateDict, Optional[Dict[int, float]]]:
        client_states, selected_clients, lr_clients = self.select_clients_and_states(local_states)
        #aggregated = self.aggregate_models(client_states, selected_clients)

        #even here i exttract primal states from local_states
        primal_states = [state['primal_state'] for state in client_states]
        aggregated = self.aggregate_models(primal_states, selected_clients)

        self.global_state = aggregated
        self.model.load_state_dict(self.global_state)
        return self.global_state, lr_clients
