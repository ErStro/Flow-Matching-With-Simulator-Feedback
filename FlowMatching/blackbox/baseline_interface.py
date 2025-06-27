import torch
import jax.numpy as jnp
from importlib.util import spec_from_file_location, module_from_spec


def load_baseline_model(
    model_path: str,
    model_file_path: str,
    *,
    dim_theta: int = 4,
    obs_dim: int | None = None,
):
    """Load the baseline flow matching network and its normalisation stats."""
    spec = spec_from_file_location("baseline_model", model_file_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    BaselineNet = module.BaselineNet

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    if obs_dim is None:
        obs_dim = checkpoint.get("obs_dim", 100)
    model = BaselineNet(dim_theta=dim_theta, obs_dim=obs_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["posterior_mean"], checkpoint["posterior_std"]


def infer_theta_hat(model, posterior_mean, posterior_std, x_obs):
    """Run a single flow-matching step to obtain \u03b8\u02c6."""
    if x_obs.ndim == 1:
        x_tensor = torch.tensor(x_obs[None, :], dtype=torch.float32)
    else:
        x_tensor = torch.tensor(x_obs, dtype=torch.float32)

    dim_theta = len(posterior_mean)
    theta_0 = torch.randn((1, dim_theta), dtype=torch.float32)
    t_tensor = torch.rand(1) * 0.2 + 0.8

    with torch.no_grad():
        v_t = model(theta_0, t_tensor, x_tensor)
        theta_t = theta_0 + t_tensor * v_t

    theta_t = theta_t.squeeze(0).numpy()
    theta_t = theta_t * posterior_std.numpy() + posterior_mean.numpy()

    return jnp.array(theta_t)
