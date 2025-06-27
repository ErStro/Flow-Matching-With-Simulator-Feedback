import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

from sbibm.tasks import get_task
import jax.numpy as jnp
import numpy as np

                                               
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

                                    
from baseline_net.flow_matching_utils import FlowMatchingUtils
from baseline_net.flow_matching_model import BaselineNet                  
try:                                             
    from .refinement_net import RefinementNet
    from .baseline_interface import load_baseline_model, infer_theta_hat
    from .lotka_volterra_loss import LotkaVolterraLoss
except ImportError:                                   
    from refinement_net import RefinementNet
    from baseline_interface import load_baseline_model, infer_theta_hat
    from lotka_volterra_loss import LotkaVolterraLoss

DEFAULT_DEBUG_LOG_PATH = os.path.join(os.path.dirname(__file__), "refinement_debug_log.txt")
DEFAULT_REFINEMENT_SAVE_PATH = os.path.join(os.path.dirname(__file__), "refinement_model.pt")


class RefinementTrainer:
    def __init__(
        self,
        baseline_model_path,
        baseline_model_file,
        epochs=1000,
        lr=1e-3,
        *,
        save_path=DEFAULT_REFINEMENT_SAVE_PATH,
        log_path=DEFAULT_DEBUG_LOG_PATH,
    ):
        self.baseline_model_path = baseline_model_path
        self.baseline_model_file = baseline_model_file
        self.epochs = epochs
        self.lr = lr
        self.save_path = save_path
        self.log_path = log_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.refine_net = RefinementNet().to(self.device)
        self.utils = FlowMatchingUtils()
        self.optimizer = optim.Adam(self.refine_net.parameters(), lr=self.lr)

                                                                             
        task = get_task("lotka_volterra")
        obs_dim = task.get_observation(1)[0].shape[0]

                                                      
        self.baseline_model, self.post_mean, self.post_std = load_baseline_model(
            self.baseline_model_path,
            self.baseline_model_file,
            obs_dim=obs_dim,
        )
        self.baseline_model.to(self.device)
        self.baseline_model.eval()

    def train(self):
        task = get_task("lotka_volterra")
        train_ids = list(range(1, 11))

        for epoch in range(self.epochs):
            total_loss = 0.0
            for idx in train_ids:
                x_obs = task.get_observation(idx)[0].numpy()
                x_obs_jnp = jnp.array(x_obs)

                theta_hat = infer_theta_hat(
                    self.baseline_model,
                    self.post_mean,
                    self.post_std,
                    x_obs,
                )

                lv = LotkaVolterraLoss(x_obs_jnp)
                x_sim = lv.simulate(theta_hat)[:10, :]
                                                                
                x_obs_cut = jnp.stack([x_obs_jnp[:10], x_obs_jnp[10:]], axis=-1)
                mse = jnp.mean((x_sim - x_obs_cut) ** 2)
                grad_jnp = lv.gradient(theta_hat)

                                                  
                t = torch.rand(1) * 0.2 + 0.8               
                theta_hat_np = np.array(theta_hat)
                if np.any(theta_hat_np <= 0):
                    print(
                        f"⚠️  Epoch {epoch+1}, Obs {idx}: theta_hat enthält nicht-positive Werte → geklammert"
                    )
                    theta_hat_np = np.clip(theta_hat_np, 0.01, None)
                theta_hat_t = torch.tensor(theta_hat_np, dtype=torch.float32).unsqueeze(0)
                grad_np = np.array(grad_jnp)
                grad_tensor = torch.tensor(grad_np, dtype=torch.float32).unsqueeze(0) / 1000.0
                mse_tensor = torch.tensor([[float(mse)]], dtype=torch.float32) / 10000.0
                if torch.isnan(mse_tensor).any() or torch.isnan(grad_tensor).any():
                    continue
                cost = torch.cat([mse_tensor, grad_tensor], dim=1)         

                                                    
                theta_1_sample = task.get_reference_posterior_samples(idx)
                theta_1_np = theta_1_sample[np.random.choice(len(theta_1_sample))]                 
                theta_1 = torch.tensor(theta_1_np, dtype=torch.float32).unsqueeze(0)
                u_t = self.utils.ot_vector_field(
                    self.utils.ot_conditional_flow(theta_hat_t, theta_1, t), theta_1, t
                )

                                                 
                v_pred = self.refine_net(t, theta_hat_t, cost)

                         
                loss = self.utils.flow_matching_loss(v_pred, u_t, t)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                with open(self.log_path, "a") as f:
                    f.write(f"Epoch {epoch+1}, Obs {idx}\n")
                    f.write(f"Theta_hat: {theta_hat}\n")
                    f.write(f"Gradient: {grad_tensor}\n")
                    f.write(f"MSE: {mse_tensor}\n")
                    f.write(f"Loss {loss}, Obs {idx}\n")
                    f.write("-" * 60 + "\n")

            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.4f}")
        torch.save(self.refine_net.state_dict(), self.save_path)

        print(f"Refinement-Netz gespeichert unter {self.save_path}")


if __name__ == "__main__":
    trainer = RefinementTrainer(
        baseline_model_path="../baseline_net/baseline_model.pt",
        baseline_model_file="../baseline_net/flow_matching_model.py",
        epochs=50
    )
    trainer.train()
