import torch
import torch.nn as nn

class Pointmaze_MLP(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 128),  # 2D input + 1D time = 3
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x, t):
        """
        x: [B, T, 2]
        t: [B]
        """
        B, T, D = x.shape  # D should be 2

        # Expand t to shape [B, T, 1] so it can be concatenated with x
        t_proj = t.view(B, 1, 1).expand(-1, T, 1)  # [B, T, 1]

        # Concatenate time to x → [B, T, 3]
        x_input = torch.cat([x, t_proj], dim=-1)  # [B, T, 3]

        # Flatten batch and time: [B*T, 3]
        x_input_flat = x_input.view(B * T, -1)

        # Pass through MLP
        out_flat = self.net(x_input_flat)  # [B*T, 2]

        # Reshape back to [B, T, 2]
        out = out_flat.view(B, T, D)
        return out
