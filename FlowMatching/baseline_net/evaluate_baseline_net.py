import torch
import matplotlib.pyplot as plt
import corner
from sbibm.tasks import get_task

from .flow_matching_model import BaselineNet
from .flow_matching_inference import PosteriorSampler


def evaluate_baseline_net(model_path,desired_obs):
    task = get_task("lotka_volterra")
    dim_theta = 4
    obs_dim = task.get_observation(1).shape[1]

                                  
    checkpoint = torch.load(model_path)
    posterior_mean = checkpoint["posterior_mean"]
    posterior_std = checkpoint["posterior_std"]

    model = BaselineNet(dim_theta=dim_theta, obs_dim=obs_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prior = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(dim_theta), torch.ones(dim_theta)), 1
    )

                          
    sampler = PosteriorSampler(model=model, prior=prior)
    x_obs = task.get_observation(desired_obs)
    with torch.no_grad():
        samples = sampler.sample(x_obs=x_obs, n_samples=1000, n_steps=10)
    samples = samples * posterior_std + posterior_mean

                       
    ref_posterior = task.get_reference_posterior_samples(1)
    ref_posterior = (ref_posterior - posterior_mean) / posterior_std

    ref_posterior = task.get_reference_posterior_samples(desired_obs)

    fig = corner.corner(
        ref_posterior.numpy(),
        color="blue",
        labels=[r"$\theta_0$", r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"],
        bins=30,
        plot_density=True,
        plot_contours=True,
        show_titles=True,
        title_fmt=".2f",
    )

    corner.corner(
        samples.numpy(),
        color="orange",
        fig=fig,                                              
        bins=30,
        plot_density=True,
        plot_contours=True,
    )

    plt.suptitle("Posteriorvergleich: Referenz (blau) vs. Flow-Matching (orange)", fontsize=14)
    plt.show()


if __name__ == "__main__":
    evaluate_baseline_net("baseline_model.pt",1)
