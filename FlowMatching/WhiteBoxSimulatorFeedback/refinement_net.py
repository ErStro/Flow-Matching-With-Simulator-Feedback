                   
import torch
import torch.nn as nn


class RefinementNet(nn.Module):
    def __init__(self, dim_theta=4, cost_dim=5, hidden_dim=128):
        super().__init__()
        self.input_dim = 1 + dim_theta + cost_dim                                       
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_theta)
        )

    def forward(self, t, theta_hat_t, cost):
        t = t.view(-1, 1)                 
        x = torch.cat([t, theta_hat_t, cost], dim=1)                                 
        return self.net(x)
