import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow_matching_inference import Batch

class FlowMatchingUtils:
    def __init__(self, sigma_min: float = 0.1):
        self.sigma_min = sigma_min

    def ot_conditional_flow(self, x0, x1, t):
        return (1 - (1 - self.sigma_min) * t)[:, None] * x0 + t[:, None] * x1

    def ot_vector_field(self, theta_t: torch.Tensor, theta_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        numerator = theta_1 - (1 - self.sigma_min) * theta_t
        denominator = 1 - (1 - self.sigma_min) * t
        return numerator / denominator[:, None]

    def draw_batch(self, prior, posterior, x_obs_tensor, batch_size: int) -> Batch:
        ids = torch.randint(0, posterior.shape[0], (batch_size,))
        theta_1 = posterior[ids, :4]
        obs_idx = posterior[ids, 4].long() - 1
        x_obs = x_obs_tensor[obs_idx]
        theta_0 = prior.sample((batch_size,))
        t = torch.rand(batch_size)
        theta_t = self.ot_conditional_flow(theta_0, theta_1, t)
        u_t = self.ot_vector_field(theta_t, theta_1, t)
        return Batch(theta_1=theta_1, theta_t=theta_t, t=t, x_obs=x_obs, u_t=u_t)

    def draw_training_batch(self, prior, posterior_all, x_obs_tensor, batch_size: int) -> Batch:
        return self.draw_batch(prior, posterior_all, x_obs_tensor, batch_size)

    def flow_matching_loss(self, pred_v: torch.Tensor, u_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t                        
        return torch.mean((pred_v - u_t) ** 2)
