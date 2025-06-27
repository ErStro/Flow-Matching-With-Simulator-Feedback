import os
import sys
import torch
import torch.optim as optim
import numpy as np
from sbibm.tasks import get_task
import jax.numpy as jnp

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from baseline_net.flow_matching_utils import FlowMatchingUtils
from WhiteBoxSimulatorFeedback.lotka_volterra_loss import LotkaVolterraLoss
try:                                       
    from .baseline_interface import load_baseline_model, infer_theta_hat
    from .encoder import ObservationEncoder
    from .finetuning_net import BlackBoxRefinementNet
except ImportError:                                   
    from baseline_interface import load_baseline_model, infer_theta_hat
    from encoder import ObservationEncoder
    from finetuning_net import BlackBoxRefinementNet

DEFAULT_DEBUG_LOG_PATH = os.path.join(os.path.dirname(__file__), "blackbox_debug_log.txt")
DEFAULT_MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "blackbox_model.pt")


class BlackBoxTrainer:
    def __init__(
        self,
        baseline_model_path,
        baseline_model_file,
        epochs=1000,
        lr=1e-3,
        *,
        save_path=DEFAULT_MODEL_SAVE_PATH,
        log_path=DEFAULT_DEBUG_LOG_PATH,
    ):
        self.baseline_model_path = baseline_model_path
        self.baseline_model_file = baseline_model_file
        self.epochs = epochs
        self.lr = lr
        self.save_path = save_path
        self.log_path = log_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = ObservationEncoder().to(self.device)
        self.refine_net = BlackBoxRefinementNet().to(self.device)
        self.utils = FlowMatchingUtils()
        params = list(self.encoder.parameters()) + list(self.refine_net.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr)

        obs_dim = get_task("lotka_volterra").get_observation(1).shape[1]
        self.baseline_model, self.post_mean, self.post_std = load_baseline_model(
            self.baseline_model_path,
            self.baseline_model_file,
            obs_dim=obs_dim,
        )

    def train(self):
        task = get_task("lotka_volterra")
        train_ids = list(range(1, 11))

        for epoch in range(self.epochs):
            total_loss = 0.0
            for idx in train_ids:
                x_obs = task.get_observation(idx)[0].numpy()
                x_obs_jnp = jnp.array(x_obs)

                theta_hat = infer_theta_hat(self.baseline_model, self.post_mean, self.post_std, x_obs)

                                          
                theta_hat_np = np.array(theta_hat)
                theta_hat_np = np.clip(theta_hat_np, a_min=1e-3, a_max=None)

                                     
                theta_hat_t = torch.tensor(theta_hat_np, dtype=torch.float32).unsqueeze(0).to(self.device)

                                                          
                lv = LotkaVolterraLoss(x_obs_jnp)
                x_sim = lv.simulate(theta_hat_np)[:10, :]

                x_obs_cut = jnp.stack([x_obs_jnp[10:], x_obs_jnp[:10]], axis=-1)

                x_sim_tensor = torch.tensor(np.array(x_sim).flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)
                x_obs_tensor = torch.tensor(np.array(x_obs_cut).flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)

                control = self.encoder(x_sim_tensor, x_obs_tensor)

                t = torch.rand(1).to(self.device) * 0.2 + 0.8

                theta_1_sample = task.get_reference_posterior_samples(idx)
                theta_1_np = theta_1_sample[np.random.choice(len(theta_1_sample))]
                theta_1 = torch.tensor(theta_1_np, dtype=torch.float32).unsqueeze(0).to(self.device)

                u_t = self.utils.ot_vector_field(
                    self.utils.ot_conditional_flow(theta_hat_t, theta_1, t),
                    theta_1,
                    t,
                )

                v_pred = self.refine_net(t, theta_hat_t, control)
                loss = self.utils.flow_matching_loss(v_pred, u_t, t)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                with open(self.log_path, "a") as f:
                    f.write(f"Epoch {epoch+1}, Obs {idx}\n")
                    f.write(f"Theta_hat: {theta_hat}\n")
                    f.write(f"Loss {loss}\n")
                    f.write("-" * 60 + "\n")

            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.4f}")

        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "refine_net": self.refine_net.state_dict(),
            },
            self.save_path,
        )
        print(f"Blackbox refinement model saved to {self.save_path}")


if __name__ == "__main__":
    

    baseline_model_file = os.path.join(os.path.dirname(__file__), "../baseline_net/flow_matching_model.py")

    trainer = BlackBoxTrainer(
        baseline_model_path = os.path.join(os.path.dirname(__file__), "..", "baseline_net", "baseline_model.pt"),
        baseline_model_file = os.path.join(os.path.dirname(__file__), "..", "baseline_net", "flow_matching_model.py"),
        epochs=50,
    )
    trainer.train()
