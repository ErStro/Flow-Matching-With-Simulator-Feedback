"""
flow_matching_training.py

Beinhaltet:
- Die Klasse `Trainer`, die ein Flow-Matching-Modell auf Basis von synthetisch generierten OT-Batches trainiert.
- Logging und Visualisierung der Gewichtsentwicklung über die Epochen.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .flow_matching_utils import FlowMatchingUtils



class Trainer:
    def __init__(
        self,
        model,
        prior,
        posterior_all,
        x_obs_tensor,
        *,
        batch_size=32,
        n_epochs=500,
        learning_rate=1e-5,
    ):
        self.model = model
        self.prior = prior
        self.posterior_all = posterior_all
        self.x_obs_tensor = x_obs_tensor
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.optim = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad], lr=learning_rate
        )
        self.prev_params = [p.data.clone() for p in self.model.parameters()]
        self.weight_drift_log = []
        self.cosine_sim_log = []

    def measure_weight_drift(self, prev_params):
        """Summiere L2-Normen der Parameteränderung über alle Layer."""
        return sum(
            torch.norm(p.data - p_prev).item()
            for p, p_prev in zip(self.model.parameters(), prev_params)
        )

    def cosine_weight_similarity(self, prev_params):
        """Berechne Cosine Similarity zwischen allen Gewichten."""
        flat_now = torch.cat([p.data.flatten() for p in self.model.parameters()])
        flat_prev = torch.cat([p_prev.flatten() for p_prev in prev_params])
        return F.cosine_similarity(flat_now.unsqueeze(0), flat_prev.unsqueeze(0)).item()

    def plot_metrics(self):
        """Plotte Weight Drift und Cosine Similarity über Epochen."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.weight_drift_log, label="Weight Drift (L2)", color="tab:blue")
        ax2.plot(self.cosine_sim_log, label="Cosine Similarity", color="tab:orange")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("L2 Drift", color="tab:blue")
        ax2.set_ylabel("Cosine Sim", color="tab:orange")
        ax1.set_title("Parameteränderung über Training")
        ax1.grid(True)
        fig.legend(loc="lower right")
        plt.tight_layout()
                    

    def train(self) -> None:
        utils = FlowMatchingUtils()
        for epoch in range(1, self.n_epochs + 1):
            batch = utils.draw_batch(
                self.prior, self.posterior_all, self.x_obs_tensor, self.batch_size
            )
            pred_v = self.model(batch.theta_t, batch.t, batch.x_obs)
            loss = utils.flow_matching_loss(pred_v, batch.u_t, batch.t)
            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optim.step()

            drift = self.measure_weight_drift(self.prev_params)
            cosine = self.cosine_weight_similarity(self.prev_params)
            self.weight_drift_log.append(drift)
            self.cosine_sim_log.append(cosine)
            self.prev_params = [p.data.clone() for p in self.model.parameters()]

            val_loss = None
            if epoch % 10 == 0:
                with torch.no_grad():
                    val_batch = utils.draw_batch(
                        self.prior, self.posterior_all, self.x_obs_tensor, self.batch_size
                    )
                    val_pred_v = self.model(val_batch.theta_t, val_batch.t, val_batch.x_obs)
                    val_loss = utils.flow_matching_loss(val_pred_v, val_batch.u_t, val_batch.t)

            if epoch % 100 == 0 or epoch == 1 or val_loss is not None:
                msg = (
                    f"Epoch {epoch:04d}/{self.n_epochs} │ "
                    f"Loss = {loss.item():.6f} │ "
                    f"Drift = {drift:.4f} │ CosSim = {cosine:.4f}"
                )
                if val_loss is not None:
                    msg += f" │ Val = {val_loss.item():.6f}"
                print(msg)

        self.plot_metrics()
