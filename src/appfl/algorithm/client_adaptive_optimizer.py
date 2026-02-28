import os
import copy
import time
import torch
import numpy as np
from .fl_base import BaseClient
from .step_size_mixins import (
    check_armijo_conditions_nls,
    try_sgd_update,
    compute_grad_norm,
    random_seed_torch,
)


class ClientAdaptOptim(BaseClient):
    """Client with local SASS (Stochastic Armijo Step Search) optimisation.

    Each communication round the client:
      1. Loads the global model sent by the server.
      2. Resets the step-size to ``server_lr`` (the configured starting LR).
      3. Runs ``num_local_epochs`` epochs over its local data; every
         *mini-batch* is one SASS step (tentative SGD update → Armijo
         check → accept / reject + step-size adjustment).
      4. Returns its updated model weights (state dict) to the server.

    SASS semantics are identical to the centralized reference in
    ``Sass/sass.py`` + ``Sass/utils.py``.
    """

    def __init__(self, id, weight, model, loss_fn, dataloader, cfg, outfile,
                 test_dataloader, metric, **kwargs):
        super(ClientAdaptOptim, self).__init__(
            id, weight, model, loss_fn, dataloader, cfg, outfile,
            test_dataloader, metric,
        )
        self.__dict__.update(kwargs)

        # ── SASS hyper-parameters (arrive via cfg.fed.args / kwargs) ────
        # Provide safe defaults so the client works even when a param is
        # missing from the config (e.g. theta was not set before this
        # refactor).
        if not hasattr(self, "server_lr"):
            self.server_lr = 0.01
        if not hasattr(self, "gamma"):       # gamma_incr
            self.gamma = 1.25
        if not hasattr(self, "gamma_decr"):
            self.gamma_decr = 0.7
        if not hasattr(self, "theta"):
            self.theta = 0.2
        if not hasattr(self, "eps_f"):
            self.eps_f = 0.0
        if not hasattr(self, "max_lr"):
            self.max_lr = 10.0
        if not hasattr(self, "min_lr"):
            self.min_lr = 1e-8

        if hasattr(self, "outfile") and self.outfile:
            self.client_log_title()

    # ── Logging ─────────────────────────────────────────────────────────

    def client_log_title(self):
        title = "%10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n" % (
            "Round", "LocalEpoch", "PerIter[s]",
            "TrainLoss", "TrainAccu",
            "TestLoss", "TestAccu",
            "StepSize", "AcceptRate", "GradNorm",
        )
        self.outfile.write(title)
        self.outfile.flush()

    def client_log_content(
        self, t, per_iter_time, train_loss, train_accuracy,
        test_loss, test_accuracy, step_size, accept_rate, grad_norm,
    ):
        contents = "%10s %10s %10.2f %10.4f %10.4f %10.4f %10.4f %10.6f %10.4f %10.4f\n" % (
            self.round, t, per_iter_time,
            train_loss, train_accuracy,
            test_loss, test_accuracy,
            step_size, accept_rate, grad_norm,
        )
        self.outfile.write(contents)
        self.outfile.flush()

    # ── Local SASS training ─────────────────────────────────────────────

    def update(self, global_model):
        """Run local SASS epochs and return the updated model state dict."""
        self.model.load_state_dict(global_model)
        self.model.to(self.cfg.device)

        # Reset step-size at the beginning of every round (requirement #7)
        step_size = float(self.server_lr)

        for t in range(self.num_local_epochs):
            start_time = time.time()
            train_loss = 0.0
            target_true, target_pred = [], []
            n_batches = 0
            n_accepted = 0
            epoch_grad_norm = 0.0

            for data, target in self.dataloader:
                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)
                n_batches += 1

                # ── deterministic seed (same as centralized sass.py) ──
                seed = int(time.time())

                # 1. Forward + backward with deterministic seed
                self.model.zero_grad()
                with random_seed_torch(seed):
                    output = self.model(data)
                    loss = self.loss_fn(output, target)
                loss.backward()

                # 2. Snapshot current parameters and gradients
                params = list(self.model.parameters())
                params_current = [p.data.clone() for p in params]
                grad_current = [
                    p.grad.clone() if p.grad is not None
                    else torch.zeros_like(p)
                    for p in params
                ]

                grad_norm = compute_grad_norm(grad_current)
                epoch_grad_norm += grad_norm.item()

                with torch.no_grad():
                    # 3. Tentative SGD step
                    try_sgd_update(params, step_size, params_current,
                                   grad_current)

                    # 4. Loss at tentative point (deterministic, same batch)
                    with random_seed_torch(seed):
                        output_next = self.model(data)
                        loss_next = self.loss_fn(output_next, target)

                    # 5. Armijo condition (faithful to centralized utils.py)
                    success = check_armijo_conditions_nls(
                        step_size=step_size,
                        loss=loss,
                        grad_norm=grad_norm,
                        loss_next=loss_next,
                        theta=self.theta,
                        eps_f=self.eps_f,
                    )

                    if success:
                        # Accept step; increase step-size
                        step_size = min(step_size * self.gamma, self.max_lr)
                        n_accepted += 1
                    else:
                        # Reject step; decrease step-size and restore params
                        step_size = max(step_size * self.gamma_decr,
                                        self.min_lr)
                        for p, pc in zip(params, params_current):
                            p.data = pc

                # Accumulate metrics (use the *accepted* loss for logging)
                train_loss += loss.item()
                target_true.append(target.detach().cpu().numpy())
                target_pred.append(output.detach().cpu().numpy())

                if self.clip_grad or self.use_dp:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_value,
                        norm_type=self.clip_norm,
                    )

            train_loss /= max(n_batches, 1)
            accept_rate = n_accepted / max(n_batches, 1)
            avg_grad_norm = epoch_grad_norm / max(n_batches, 1)

            target_true = np.concatenate(target_true)
            target_pred = np.concatenate(target_pred)
            train_accuracy = float(self.metric(target_true, target_pred))

            # Validation
            if self.cfg.validation and self.test_dataloader is not None:
                test_loss, test_accuracy = super(
                    ClientAdaptOptim, self
                ).client_validation()
            else:
                test_loss, test_accuracy = 0, 0

            per_iter_time = time.time() - start_time

            self.client_log_content(
                t + 1, per_iter_time, train_loss, train_accuracy,
                test_loss, test_accuracy, step_size, accept_rate,
                avg_grad_norm,
            )

            # Save model state dict if requested
            if self.cfg.save_model_state_dict:
                path = self.cfg.output_dirname + f"/client_{self.id}"
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(path, f"{self.round}_{t}.pt"),
                )

        self.round += 1

        # ── Prepare return value: plain state dict (weights only) ───────
        self.primal_state = copy.deepcopy(self.model.state_dict())

        # Differential privacy (if configured)
        if self.use_dp:
            sensitivity = 2.0 * self.clip_value * step_size
            scale_value = sensitivity / self.epsilon
            super(ClientAdaptOptim, self).laplace_mechanism_output_perturb(
                scale_value
            )

        return self.primal_state
