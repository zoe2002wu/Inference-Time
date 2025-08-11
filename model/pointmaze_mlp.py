class Pointmaze_MLP(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, 128), 
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x, t):
        t_proj = t.expand(-1, x.shape[1])  # Shape: [B, 2]
        x_input = torch.cat([x, t_proj], dim=1)  # Shape: [B, 4]
        return self.net(x_input)