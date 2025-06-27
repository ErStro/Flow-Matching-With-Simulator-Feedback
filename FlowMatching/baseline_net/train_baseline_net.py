import torch
from sbibm.tasks import get_task
from .flow_matching_model import BaselineNet
from .flow_matching_training import Trainer


class BaselineNetTrainer:
    def __init__(self, task_name="lotka_volterra", save_path="baseline_model.pt", epochs=1000):
        self.task_name = task_name
        self.save_path = save_path
        self.epochs = epochs
        self.model = None
        self.posterior_mean = None
        self.posterior_std = None

    def _prepare_data(self):
        torch.manual_seed(42)
        task = get_task(self.task_name)
        train_ids = list(range(1, 11))

        posterior_all = torch.cat([
            torch.cat([
                task.get_reference_posterior_samples(i),
                torch.full((task.get_reference_posterior_samples(i).shape[0], 1), i, dtype=torch.int32)
            ], dim=1) for i in train_ids
        ], dim=0)

        dim_theta = 4
        self.posterior_mean = posterior_all[:, :dim_theta].mean(0)
        self.posterior_std = posterior_all[:, :dim_theta].std(0)
        posterior_all[:, :dim_theta] = (posterior_all[:, :dim_theta] - self.posterior_mean) / self.posterior_std

        x_obs_tensor = torch.cat([task.get_observation(i) for i in train_ids], dim=0)
        obs_dim = x_obs_tensor.shape[1]

        prior = torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros(dim_theta), torch.ones(dim_theta)), 1
        )

        return posterior_all, x_obs_tensor, dim_theta, obs_dim, prior

    def train(self, *, save_path=None):
        posterior_all, x_obs_tensor, dim_theta, obs_dim, prior = self._prepare_data()

        self.model = BaselineNet(dim_theta=dim_theta, obs_dim=obs_dim)
        trainer = Trainer(
            model=self.model,
            prior=prior,
            posterior_all=posterior_all,
            x_obs_tensor=x_obs_tensor,
            batch_size=32,
            n_epochs=self.epochs,
            learning_rate=1e-3,
        )
        trainer.train()

        if save_path is None:
            save_path = self.save_path

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "posterior_mean": self.posterior_mean,
                "posterior_std": self.posterior_std,
                "obs_dim": obs_dim,
            },
            save_path,
        )

        print(f"Modell gespeichert unter {save_path}")


if __name__ == "__main__":
    trainer = BaselineNetTrainer(save_path="baseline_model.pt", epochs=2000)
    trainer.train()
