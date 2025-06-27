import torch
import torch.nn as nn

class BlackBoxRefinementNet(nn.Module):
    """Refine theta_hat in the last part of the flow."""
    def __init__(self, dim_theta=4, control_dim=16, hidden_dim=128):
        super().__init__()
        self.input_dim = 1 + dim_theta + control_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_theta),
        )

    def forward(self, t: torch.Tensor, theta_hat: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        x = torch.cat([t, theta_hat, control], dim=1)
        return self.net(x)
