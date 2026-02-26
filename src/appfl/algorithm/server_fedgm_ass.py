from __future__ import annotations
from typing import Any, Dict, List, Tuple
import torch

from .server_fed_avg import ServerFedAvg
from .aggregation_mixins import GeometricMedianMixin
from .step_size_mixins import AdaptiveMixin

StateDict = Dict[str, torch.Tensor]

class FedGMASS(AdaptiveMixin, GeometricMedianMixin, ServerFedAvg):
    """Geometric Median with Adaptive Step Size Selection (Algorithm 1 from PDF).
    
    Uses:
    - AdaptiveMixin for client selection and gradient-based updates
    - GeometricMedianMixin for aggregation (though AdaptiveMixin handles the main update)
    - ServerFedAvg base for the overall FL framework
    """

    def update(self, local_states: List[Any]) -> Tuple[StateDict, Dict[int, float]]:
        """
        Server update following Algorithm 1.
        
        AdaptiveMixin.select_clients_and_states will:
        1. Select clients based on sufficient decrease condition (Eq 5 or 6)
        2. Compute new global state using gradient-based update (line 11)
        3. Update per-client learning rates (line 12)
        
        Returns the new global state and per-client learning rates.
        """
        # AdaptiveMixin returns [new_global_state], selected_clients, lr_clients
        # where new_global_state is already the updated model (wᵏ⁺¹)
        _, selected_clients, lr_clients = self.select_clients_and_states(local_states)
        
        """
        # Extract the single aggregated state
        self.global_state = aggregated_states[0]
        self.model.load_state_dict(self.global_state)
        """

        #my fix: compute geometric median on the selected clients' states
        #ClientAdaptOptim.update return a dict like this: 
        """
            "primal_state": self.primal_state,
            "grad_estimate": self.grad_estimate,
            "function_value_difference": self.function_value_difference
        }
        """
        #in run_serial_adapt before calling server.update we add each client state to the local_state list
        #so local_states is a list of dicts, each dict has the keys "primal_state", "grad_estimate", "function_value_difference"
        #aggregate models requires a list of STATEDICT
        #so we need to extract the "primal_state" from each client state and pass it to aggregate_models
        primal_states = [client_state["primal_state"] for client_state in local_states]
        self.global_state = self.aggregate_models(primal_states,selected_clients)
        self.model.load_state_dict(self.global_state)
        
        return self.global_state, lr_clients
