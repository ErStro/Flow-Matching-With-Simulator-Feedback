import torch
import jax.numpy as jnp
from .lotka_volterra_loss import LotkaVolterraLoss


def load_baseline_model(model_path, model_file_path, dim_theta=4, obs_dim=100):
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("baseline_model", model_file_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    BaselineNet = module.BaselineNet

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model = BaselineNet(dim_theta=dim_theta, obs_dim=obs_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["posterior_mean"], checkpoint["posterior_std"]


def infer_theta_hat(model, posterior_mean, posterior_std, x_obs):
                               
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


def evaluate_baseline_with_sim(model_path, model_file_path, x_obs):
                                     
    model, mean, std = load_baseline_model(model_path, model_file_path, obs_dim=x_obs.shape[0])

                 
    theta_hat = infer_theta_hat(model, mean, std, x_obs)

                                    
    x_obs_jnp = jnp.array(x_obs)
    lv = LotkaVolterraLoss(x_obs_jnp)

                                                                             
    x_sim = lv.simulate(theta_hat)[:10, :]                         
                                                                                
    x_obs_cut = jnp.stack([x_obs_jnp[:10], x_obs_jnp[10:]], axis=-1)


                                
    mse = jnp.mean((x_sim - x_obs_cut) ** 2)
    loss = lv.loss(theta_hat)
    gradient = lv.gradient(theta_hat)

    return mse, theta_hat, x_sim, loss, gradient

