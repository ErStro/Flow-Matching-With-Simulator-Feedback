import os
import numpy as np
import torch
import jax.numpy as jnp
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KernelDensity

from sbibm.tasks import get_task

from baseline_net.flow_matching_inference import PosteriorSampler
from WhiteBoxSimulatorFeedback.baseline_interface import load_baseline_model, infer_theta_hat
from WhiteBoxSimulatorFeedback.refinement_net import RefinementNet
from WhiteBoxSimulatorFeedback.lotka_volterra_loss import LotkaVolterraLoss
from blackbox.finetuning_net import BlackBoxRefinementNet
from blackbox.encoder import ObservationEncoder


                                                                               
                  
                                                                               

def c2st(X: torch.Tensor, Y: torch.Tensor, seed: int = 1, n_folds: int = 5, z_score: bool = True) -> torch.Tensor:
    """Classifier two-sample test using accuracy."""
    if z_score:
        mean = X.mean(0)
        std = X.std(0)
        X = (X - mean) / std
        Y = (Y - mean) / std

    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    data = np.concatenate([X_np, Y_np])
    target = np.concatenate([np.zeros(len(X_np)), np.ones(len(Y_np))])

    clf = MLPClassifier(hidden_layer_sizes=(10 * X.shape[1], 10 * X.shape[1]), max_iter=10000, solver="adam", random_state=seed)
    n_folds = max(2, min(n_folds, len(data)))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=kf, scoring="accuracy")
    return torch.tensor([scores.mean()], dtype=torch.float32)


def mmd_rbf(X: torch.Tensor, Y: torch.Tensor, subset: int = 1000) -> torch.Tensor:
    """Compute MMD^2 with RBF kernel using a subset for efficiency."""
    if X.shape[0] > subset:
        idx = torch.randperm(X.shape[0])[:subset]
        X = X[idx]
    if Y.shape[0] > subset:
        idx = torch.randperm(Y.shape[0])[:subset]
        Y = Y[idx]
    XY = torch.cat([X, Y], dim=0)
    sigma = torch.median(torch.pdist(XY))
    gamma = 1.0 / (2.0 * sigma ** 2)
    Kxx = torch.exp(-gamma * torch.cdist(X, X) ** 2).mean()
    Kyy = torch.exp(-gamma * torch.cdist(Y, Y) ** 2).mean()
    Kxy = torch.exp(-gamma * torch.cdist(X, Y) ** 2).mean()
    return Kxx + Kyy - 2 * Kxy


def nll_kde(samples: np.ndarray, ref: np.ndarray) -> float:
    """Estimate negative log likelihood of ref under KDE fitted to samples."""
    n, d = samples.shape
    std = samples.std(0).mean()
    bandwidth = std * n ** (-1.0 / (d + 4))
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(samples)
    log_probs = kde.score_samples(ref)
    return float(-log_probs.mean())


                                                                               
                  
                                                                               


def sample_baseline(task, model_path, model_file, obs_idx, n_samples=10, n_steps=10):
    x_obs = task.get_observation(obs_idx)[0]
    obs_dim = x_obs.shape[0]
    baseline, mean, std = load_baseline_model(model_path, model_file, obs_dim=obs_dim)
    prior = torch.distributions.Independent(torch.distributions.Normal(torch.zeros(4), torch.ones(4)), 1)
    sampler = PosteriorSampler(model=baseline, prior=prior)
    with torch.no_grad():
        z = sampler.sample(x_obs=x_obs, n_samples=n_samples, n_steps=n_steps)
    return z * std + mean


def sample_whitebox(task, baseline_path, baseline_file, refine_path, obs_idx, n_samples=10, n_steps=10):
    x_obs = task.get_observation(obs_idx)[0].numpy()
    obs_dim = x_obs.shape[0]
    baseline, mean, std = load_baseline_model(baseline_path, baseline_file, obs_dim=obs_dim)
    refine_net = RefinementNet()
    refine_net.load_state_dict(torch.load(refine_path, map_location="cpu"))
    refine_net.eval()
    prior = torch.distributions.Independent(torch.distributions.Normal(torch.zeros(4), torch.ones(4)), 1)
    sampler = PosteriorSampler(model=baseline, prior=prior)
    lv = LotkaVolterraLoss(jnp.array(x_obs))

    refined = []
    with torch.no_grad():
        for _ in range(n_samples):
            z = sampler.sample(x_obs=torch.tensor(x_obs, dtype=torch.float32), n_samples=1, n_steps=n_steps)[0]
            theta = z * std + mean
            theta_np = np.clip(theta.numpy(), 0.01, None)

            x_sim = lv.simulate(theta_np)[:10, :]
            x_obs_cut = jnp.stack([x_obs[:10], x_obs[10:]], axis=-1)
            mse = jnp.mean((x_sim - x_obs_cut) ** 2)
            grad = lv.gradient(theta_np)
            cost = torch.cat([
                torch.tensor([[float(mse)]], dtype=torch.float32) / 10000.0,
                torch.tensor(np.array(grad), dtype=torch.float32).unsqueeze(0) / 1000.0,
            ], dim=1)
            t = torch.rand(1)
            delta = refine_net(t, theta.unsqueeze(0), cost).squeeze(0)
            refined.append((theta + delta).numpy())
    return torch.tensor(np.stack(refined), dtype=torch.float32)


def sample_blackbox(task, baseline_path, baseline_file, blackbox_path, obs_idx, n_samples=10):
    x_obs = task.get_observation(obs_idx)[0].numpy()
    obs_dim = x_obs.shape[0]
    baseline, mean, std = load_baseline_model(baseline_path, baseline_file, obs_dim=obs_dim)
    state = torch.load(blackbox_path, map_location="cpu")
    encoder = ObservationEncoder()
    refine_net = BlackBoxRefinementNet()
    encoder.load_state_dict(state["encoder"])
    refine_net.load_state_dict(state["refine_net"])
    encoder.eval()
    refine_net.eval()
    lv = LotkaVolterraLoss(jnp.array(x_obs))

    refined = []
    with torch.no_grad():
        for _ in range(n_samples):
            theta_hat = infer_theta_hat(baseline, mean, std, x_obs)
            theta_np = np.clip(np.array(theta_hat), a_min=1e-3, a_max=None)
            theta_t = torch.tensor(theta_np, dtype=torch.float32).unsqueeze(0)
            x_sim = lv.simulate(theta_np)[:10, :]
            x_obs_cut = jnp.stack([x_obs[10:], x_obs[:10]], axis=-1)
            x_sim_t = torch.tensor(np.array(x_sim).flatten(), dtype=torch.float32).unsqueeze(0)
            x_obs_t = torch.tensor(np.array(x_obs_cut).flatten(), dtype=torch.float32).unsqueeze(0)
            control = encoder(x_sim_t, x_obs_t)
            t_val = torch.rand(1)
            v_pred = refine_net(t_val, theta_t, control)
            theta_ref = theta_t + (1.0 - t_val) * v_pred
            refined.append(theta_ref.squeeze(0).numpy())
    return torch.tensor(np.stack(refined), dtype=torch.float32)


                                                                               
                         
                                                                               


def _filter_finite(X: torch.Tensor) -> torch.Tensor:
    """Remove rows containing NaN or Inf values."""
    mask = torch.isfinite(X).all(dim=1)
    return X[mask]


def compute_metrics(samples: torch.Tensor, ref: torch.Tensor) -> dict:
    samples = _filter_finite(samples)
    ref = _filter_finite(ref)
    return {
        "c2st": c2st(samples, ref).item(),
        "mmd": mmd_rbf(samples, ref).item(),
        "nll": nll_kde(samples.numpy(), ref.numpy()),
    }


def main():
    base_dir = os.path.dirname(__file__)
    baseline_dir = os.path.join(base_dir, "baseline_net")
    whitebox_dir = os.path.join(base_dir, "WhiteBoxSimulatorFeedback")
    blackbox_dir = os.path.join(base_dir, "blackbox")

    baseline_1000 = os.path.join(baseline_dir, "baseline_model_1000.pt")
    baseline_1500 = os.path.join(baseline_dir, "baseline_model_1500.pt")
    baseline_model_file = os.path.join(baseline_dir, "flow_matching_model.py")

    whitebox_model = os.path.join(whitebox_dir, "refinement_model_500.pt")
    blackbox_model = os.path.join(blackbox_dir, "blackbox_model_500.pt")

    task = get_task("lotka_volterra")
    num_obs = task.num_observations

    results = {"baseline": [], "whitebox": [], "blackbox": []}

    for idx in range(1, num_obs + 1):
        ref_samples = task.get_reference_posterior_samples(idx)

        baseline_samples = sample_baseline(task, baseline_1500, baseline_model_file, idx, n_samples=5)
        wb_samples = sample_whitebox(task, baseline_1000, baseline_model_file, whitebox_model, idx, n_samples=5)
        bb_samples = sample_blackbox(task, baseline_1000, baseline_model_file, blackbox_model, idx, n_samples=5)

        results["baseline"].append(compute_metrics(baseline_samples, ref_samples))
        results["whitebox"].append(compute_metrics(wb_samples, ref_samples))
        results["blackbox"].append(compute_metrics(bb_samples, ref_samples))

    for name, metrics in results.items():
        c2st_mean = np.mean([m["c2st"] for m in metrics])
        mmd_mean = np.mean([m["mmd"] for m in metrics])
        nll_mean = np.mean([m["nll"] for m in metrics])
        print(f"{name} - C2ST: {c2st_mean:.4f}, MMD: {mmd_mean:.4f}, NLL: {nll_mean:.4f}")


if __name__ == "__main__":
    main()
