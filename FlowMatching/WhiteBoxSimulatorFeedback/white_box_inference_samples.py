                                
import os
import sys
import torch
import numpy as np
import jax.numpy as jnp

                                                                        
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from baseline_net.flow_matching_inference import PosteriorSampler
from WhiteBoxSimulatorFeedback.refinement_net import RefinementNet
from WhiteBoxSimulatorFeedback.lotka_volterra_loss import LotkaVolterraLoss
from WhiteBoxSimulatorFeedback.baseline_interface import load_baseline_model


                            
BASELINE_MODEL_PATH = os.path.join(project_root, "baseline_net", "baseline_model.pt")
REFINEMENT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "refinement_model.pt")

def load_baseline(obs_dim):
    """Load baseline network and statistics."""
    model, mean, std = load_baseline_model(
        BASELINE_MODEL_PATH,
        os.path.join(project_root, "baseline_net", "flow_matching_model.py"),
        obs_dim=obs_dim,
    )
    return model, mean, std

def load_refinement():
    """Load refinement network."""
    model = RefinementNet()
    state = torch.load(REFINEMENT_MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    return model

def compute_cost(theta, x_obs):
    """Compute simulator MSE and gradient for a parameter sample."""
    x_obs_jnp = jnp.array(x_obs)
    lv = LotkaVolterraLoss(x_obs_jnp)
    x_sim = lv.simulate(theta)[:10, :]
                                                    
    x_obs_cut = jnp.stack([x_obs_jnp[:10], x_obs_jnp[10:]], axis=-1)
    mse = jnp.mean((x_sim - x_obs_cut) ** 2)
    grad = lv.gradient(theta)
    grad_np = np.array(grad)
    mse = torch.tensor([[float(mse)]], dtype=torch.float32) / 10000.0
    grad_tensor = torch.tensor(grad_np, dtype=torch.float32).unsqueeze(0) / 1000.0
    return float(mse), np.asarray(grad)

def refine_samples(x_obs, n_samples=100, n_steps=10, device="cpu"):
    """Baseline- und refined-Samples einer Beobachtung zurückgeben – ohne Skalen-Mismatch."""
                                                        
                                
                                                        
    obs_dim                    = len(x_obs)
    baseline, mean, std        = load_baseline(obs_dim)
    refine_net                 = load_refinement().to(device).eval()

    prior   = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(4, device=device),
                                   torch.ones(4,  device=device)), 1)
    sampler = PosteriorSampler(baseline.to(device), prior)

                                                        
                                                            
                                                        
    x_obs_t   = torch.tensor(x_obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        z_samples = sampler.sample(x_obs_t, n_samples=n_samples, n_steps=n_steps)
    theta_base = z_samples * torch.tensor(std, device=device) + torch.tensor(mean, device=device)
    theta_base = torch.clamp(theta_base, min=0.01)

                                                        
                                                                      
                                                        
    refined = []
    for theta in theta_base:                                        
        theta_np = theta.cpu().numpy()

        mse, grad = compute_cost(theta_np, x_obs)                                           

        t_tensor     = torch.rand(1, device=device) * 0.2 + 0.8                     
        theta_tensor = theta.unsqueeze(0)                                            
        mse_tensor   = torch.tensor([[mse]],  dtype=torch.float32, device=device)
        grad_tensor  = torch.tensor(grad,  dtype=torch.float32, device=device).unsqueeze(0)
        cost         = torch.cat([mse_tensor, grad_tensor], dim=1)       

        if not torch.isfinite(cost).all():
            print("⚠️  cost enthält NaN/Inf – Sample übersprungen"); continue

        with torch.no_grad():
            delta = refine_net(t_tensor, theta_tensor, cost).squeeze(0)

        

        if not torch.isfinite(delta).all():
            print(f"⚠️  Nicht-finite Δ – übersprungen"); continue

        refined.append((theta + delta).cpu().numpy())

    if not refined:
        raise RuntimeError("Kein gültiges refined-Sample erzeugt!")
    return theta_base.cpu().numpy(), np.stack(refined)

    refined = np.stack(refined)
    return samples.numpy(), refined

def sample_all_observations(n_samples=100, n_steps=10):
    """Sample baseline and refined posteriors for all observations."""
    from sbibm.tasks import get_task

    task = get_task("lotka_volterra")
    baseline_res = {}
    refined_res = {}
    for idx in range(1, task.num_observations + 1):
        x_obs = task.get_observation(idx)[0].numpy()
        base, ref = refine_samples(x_obs, n_samples=n_samples, n_steps=n_steps)
        print("base",base,"ref",ref)
        baseline_res[idx] = base
        refined_res[idx] = ref
    return baseline_res, refined_res


def plot_samples(samples, title):
    import corner
    import matplotlib.pyplot as plt
    samples = samples[np.isfinite(samples).all(axis=1)]
    print(f"Plotting {samples.shape[0]} valid samples")
    fig = corner.corner(
        samples,
        bins=30,
        labels=[r"$\theta_0$", r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"],
        show_titles=True,
    )
    plt.suptitle(title)
    plt.show()
    return fig


def main():
    baseline_samp, refined_samp = sample_all_observations(n_samples=10, n_steps=10)

    plot_samples(baseline_samp[5], "Baseline Samples Observation 5")
    plot_samples(refined_samp[2], "Refined Samples Observation 2")


if __name__ == "__main__":
    main()
