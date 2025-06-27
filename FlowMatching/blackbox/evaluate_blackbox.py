import os
import sys
import torch
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import corner
from sbibm.tasks import get_task

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from WhiteBoxSimulatorFeedback.lotka_volterra_loss import LotkaVolterraLoss

                                     
try:
    from .baseline_interface import load_baseline_model, infer_theta_hat
    from .encoder import ObservationEncoder
    from .finetuning_net import BlackBoxRefinementNet
except ImportError:                              
    from baseline_interface import load_baseline_model, infer_theta_hat
    from encoder import ObservationEncoder
    from finetuning_net import BlackBoxRefinementNet


def evaluate_blackbox(
    baseline_model_path: str,
    baseline_model_file: str,
    blackbox_model_path: str,
    obs_idx: int = 1,
    n_samples: int = 50,
):
    """Compare baseline and blackbox refinement for a single observation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                                                                    
                 
                                                                    
    task = get_task("lotka_volterra")
    x_obs = task.get_observation(obs_idx)[0].numpy()
    theta_true = task.get_true_parameters(obs_idx).squeeze(0).numpy()
    baseline, mean, std = load_baseline_model(
        baseline_model_path, baseline_model_file, obs_dim=x_obs.shape[0]
    )

    state = torch.load(blackbox_model_path, map_location=device)
    encoder = ObservationEncoder().to(device)
    refine_net = BlackBoxRefinementNet().to(device)
    encoder.load_state_dict(state["encoder"])
    refine_net.load_state_dict(state["refine_net"])
    encoder.eval()
    refine_net.eval()

    baseline_samples = []
    refined_samples = []

    x_obs_jnp = jnp.array(x_obs)
    lv = LotkaVolterraLoss(x_obs_jnp)

    for _ in range(n_samples):
                                                                    
        theta_hat = infer_theta_hat(baseline, mean, std, x_obs)

                                                   
        theta_np = np.array(theta_hat)
        theta_np = np.clip(theta_np, a_min=1e-3, a_max=None)

        theta_t = torch.tensor(theta_np, dtype=torch.float32).unsqueeze(0).to(device)
        baseline_samples.append(theta_np)

                                                                    
                                                 
        x_sim = lv.simulate(theta_np)[:10, :]

        x_obs_cut = jnp.stack([x_obs_jnp[10:], x_obs_jnp[:10]], axis=-1)
        x_sim_t = torch.tensor(np.array(x_sim).flatten(), dtype=torch.float32).unsqueeze(0).to(device)
        x_obs_t = torch.tensor(np.array(x_obs_cut).flatten(), dtype=torch.float32).unsqueeze(0).to(device)
        control = encoder(x_sim_t, x_obs_t)

                                                                    
        t_val = torch.rand(1).to(device) * 0.2 + 0.8
        with torch.no_grad():
            v_pred = refine_net(t_val, theta_t, control)
            theta_ref = theta_t + (1.0 - t_val) * v_pred
        refined_samples.append(theta_ref.cpu().numpy()[0])

    baseline_samples = np.array(baseline_samples)
    refined_samples = np.array(refined_samples)
    print(baseline_samples,refined_samples)

                                                                    
                 
                                                                    

    fig = corner.corner(
        baseline_samples,
        color="orange",
        truths       = theta_true,
        labels=[r"$\theta_0$", r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"],
        show_titles=True,
        title_fmt=".2f",
    )

    corner.corner(
        refined_samples,
        fig=fig,
        color="green",
        show_titles=True,
        title_fmt=".2f",
    )

    plt.suptitle(f"Baseline (orange) vs Blackbox (green) - Observation {obs_idx}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    baseline_model_file = os.path.join(os.path.dirname(__file__), "../baseline_net/flow_matching_model.py")

    evaluate_blackbox(
        baseline_model_path = os.path.join(os.path.dirname(__file__), "..", "baseline_net", "baseline_model.pt"),
        baseline_model_file = os.path.join(os.path.dirname(__file__), "..", "baseline_net", "flow_matching_model.py"),
        blackbox_model_path = os.path.join(os.path.dirname(__file__), "blackbox_model.pt"),
        obs_idx=8,
        n_samples=1000,
    )
