import torch
from dataclasses import dataclass
from torchdiffeq import odeint


@dataclass
class Batch:
    theta_1: torch.Tensor                                   
    theta_t: torch.Tensor                                          
    t:        torch.Tensor                   
    x_obs:    torch.Tensor                   
    u_t:      torch.Tensor                                      


class PosteriorSampler:
    def __init__(self, model, prior):
        """
        :param model: trainiertes Flow-Matching-Modell (z. B. BaselineNet)
        :param prior: torch-distribution mit sample() Methode
        """
        self.model = model
        self.prior = prior

    def sample(self, x_obs: torch.Tensor, n_samples: int = 10, n_steps: int = 5) -> torch.Tensor:
        """
        Integriert das Vektorfeld vom Prior zum Posterior

        :param x_obs: Beobachtung, Tensor der Form [obs_dim]
        :param n_samples: Anzahl an Samples
        :param n_steps: Anzahl Integrationsschritte (Zeitauflösung)
        :return: Posterior-Samples [n_samples, theta_dim]
        """
        x_obs = x_obs.expand(n_samples, -1).detach()
        theta_0 = self.prior.sample((n_samples,)).detach()

        def vf(t, theta):
            t_scalar = t.item()
            t_tensor = torch.full((n_samples,), t_scalar).to(theta)
            return self.model(theta, t_tensor, x_obs)

        t_vals = torch.linspace(0.0, 1.0, n_steps).to(theta_0)
        trajectory = odeint(vf, theta_0, t_vals, atol=1e-5, rtol=1e-5, method="dopri5")
        return trajectory[-1].detach()