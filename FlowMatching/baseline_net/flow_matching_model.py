import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.skip_proj = (
            nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.elu(self.linear(x))
        return self.norm(h + self.skip_proj(x))


class BaselineNet(nn.Module):
    def __init__(self, dim_theta: int, obs_dim: int):
        super().__init__()
        self.t_emb = nn.Linear(1, 32)
        self.x_emb = nn.Linear(obs_dim, 32)
        dims = [dim_theta + 64, 32, 64, 128, 256, 512, 512, 512, 512, 512, 256, 128, 64, 32, dim_theta]
        blocks = [ResidualBlock(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        self.net = nn.Sequential(*blocks)

    def forward(self, theta_t: torch.Tensor, t: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_emb(t.unsqueeze(-1))
        x_emb = self.x_emb(x_obs)
        x = torch.cat([theta_t, t_emb, x_emb], dim=-1)
        return self.net(x)
