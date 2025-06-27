import torch
import torch.nn as nn
import sys
import os
import numpy as np
from sbibm.tasks import get_task
import jax.numpy as jnp

            
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from baseline_net.flow_matching_utils import FlowMatchingUtils
from refinement_net import RefinementNet
from baseline_interface import load_baseline_model, infer_theta_hat
from lotka_volterra_loss import LotkaVolterraLoss


class RefinementEvaluator:
    def __init__(self, baseline_model_path, baseline_model_file, refinement_model_path):
        self.baseline_model_path = baseline_model_path
        self.baseline_model_file = baseline_model_file
        self.refinement_model_path = refinement_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.refine_net = RefinementNet().to(self.device)
        self.refine_net.load_state_dict(torch.load(refinement_model_path, map_location=self.device))
        self.refine_net.eval()

        self.utils = FlowMatchingUtils()

    def evaluate(self, task_name="lotka_volterra", num_obs=1, n_samples=50):
        import matplotlib.pyplot as plt
        import corner

        task = get_task(task_name)
        idx = 7                     
        x_obs = task.get_observation(idx)[0].numpy()
        posterior_samples = task.get_reference_posterior_samples(idx)
        posterior_mean = posterior_samples.mean(axis=0).numpy()

        baseline_dists = []
        refined_dists = []
        baseline_mses = []
        refined_mses = []
        baseline_samples = []
        refined_samples = []

        print(f"\n🔍 Evaluierung auf Task '{task_name}', Observation {idx}, {n_samples} Samples\n")

                                  
        baseline_model, mean, std = load_baseline_model(
            self.baseline_model_path,
            self.baseline_model_file,
            obs_dim=x_obs.shape[0],
        )

        for _ in range(n_samples):
                                            
            theta_hat = infer_theta_hat(baseline_model, mean, std, x_obs)

            x_obs_jnp = jnp.array(x_obs)
            lv = LotkaVolterraLoss(x_obs_jnp)
            x_sim = lv.simulate(theta_hat)[:10, :]
                                                            
            x_obs_cut = jnp.stack([x_obs_jnp[:10], x_obs_jnp[10:]], axis=-1)
            mse_base = jnp.mean((x_sim - x_obs_cut) ** 2)
            grad_jnp = lv.gradient(theta_hat)

            theta_hat_np = np.array(theta_hat)
            if np.any(theta_hat_np <= 0):
                theta_hat_np = np.clip(theta_hat_np, 0.01, None)
            theta_hat_t = torch.tensor(theta_hat_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            grad_np = np.array(grad_jnp)
            grad_tensor = torch.tensor(grad_np, dtype=torch.float32).unsqueeze(0).to(self.device) / 1000.0
            mse_tensor = torch.tensor([[float(mse_base)]], dtype=torch.float32).to(self.device) / 10000.0

            if torch.isnan(mse_tensor).any() or torch.isnan(grad_tensor).any():
                continue

            cost = torch.cat([mse_tensor, grad_tensor], dim=1)          

                              
            t = torch.rand(1).to(self.device) * 0.2 + 0.8

                          
            with torch.no_grad():
                v_refine = self.refine_net(t, theta_hat_t, cost)
                theta_refined = theta_hat_t + (1.0 - t) * v_refine

            theta_refined_np = theta_refined.cpu().numpy()[0]
            x_sim_ref = lv.simulate(theta_refined_np)[:10, :]
            mse_ref = jnp.mean((x_sim_ref - x_obs_cut) ** 2)

                                                                  
            delta = torch.abs(theta_refined - theta_hat_t)
            relative_change = delta / torch.abs(theta_hat_t + 1e-8)

            if torch.any(relative_change > 0.1):
                print("⚠️  Refinement verändert theta_hat um >10% → Sample verworfen")
                continue

                            
            l2_baseline = np.linalg.norm(theta_hat_np - posterior_mean)
            l2_refined = np.linalg.norm(theta_refined.cpu().numpy()[0] - posterior_mean)

            baseline_dists.append(l2_baseline)
            refined_dists.append(l2_refined)
            baseline_mses.append(float(mse_base))
            refined_mses.append(float(mse_ref))

            baseline_samples.append(theta_hat_np)
            refined_samples.append(theta_refined_np)

                     
        mean_base = np.mean(baseline_dists)
        mean_ref = np.mean(refined_dists)
        std_base = np.std(baseline_dists)
        std_ref = np.std(refined_dists)
        mse_base_mean = np.mean(baseline_mses)
        mse_ref_mean = np.mean(refined_mses)

        print(f"\n📈 Ergebnisse für Observation {idx}")
        print(f"  ▸ ⌀ L2 Baseline     : {mean_base:.4f} ± {std_base:.4f}")
        print(f"  ▸ ⌀ L2 Refinement   : {mean_ref:.4f} ± {std_ref:.4f}")
        print(f"  ▸ Verbesserung      : {mean_base - mean_ref:.4f}\n")
        print(f"  ▸ MSE Baseline      : {mse_base_mean:.6f}")
        print(f"  ▸ MSE Refinement    : {mse_ref_mean:.6f}\n")

                       
        baseline_samples = np.array(baseline_samples)
        refined_samples = np.array(refined_samples)

        fig = corner.corner(
            baseline_samples,
            labels=[r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$"],
            color="blue",
            truths=np.mean(baseline_samples, axis=0),
            truth_color="blue",
            title_fmt=".2f",
            label_kwargs={"fontsize": 12},
            show_titles=True,
            title_kwargs={"fontsize": 10}
        )

        corner.corner(
            refined_samples,
            fig=fig,
            color="green",
            truths=np.mean(refined_samples, axis=0),
            truth_color="green",
            title_fmt=".2f",
            label_kwargs={"fontsize": 12},
            show_titles=True,
            title_kwargs={"fontsize": 10}
        )

        plt.suptitle("Corner Plot – Baseline (blau) vs Refinement (grün), Obs 5", fontsize=16)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    evaluator = RefinementEvaluator(
        baseline_model_path="../baseline_net/baseline_model.pt",
        baseline_model_file="../baseline_net/flow_matching_model.py",
        refinement_model_path="refinement_model.pt"
    )
    evaluator.evaluate(task_name="lotka_volterra")

