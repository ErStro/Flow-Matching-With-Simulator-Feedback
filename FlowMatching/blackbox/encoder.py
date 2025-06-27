import torch
import torch.nn as nn

class ObservationEncoder(nn.Module):
    """Encode simulator output and observation into a control signal."""
    def __init__(self, input_dim=40, hidden_dim=64, output_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x_sim: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
        """x_sim and x_obs expected flattened."""
        x_sim = x_sim.view(x_sim.size(0), -1)
        x_obs = x_obs.view(x_obs.size(0), -1)
        x = torch.cat([x_sim, x_obs], dim=1)
        return self.net(x)
