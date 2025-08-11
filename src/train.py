import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model.pointmaze_mlp import Pointmaze_MLP

class DiffusionDataset(Dataset):
    '''
    Parse dataset
    '''
    def __init__(self, data, compact=True):
        if compact:
            self.obs = data['observations']
            self.valids = data['valids']
            self.obs = self.obs[self.valids.astype(bool)]
        else:
            self.obs = data['observations']
        self.obs = torch.tensor(self.obs, dtype=torch.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx]

def q_sample(x0, t):
    '''
    Forward Process
    '''
    noise = torch.randn_like(x0)
    alpha_t = 1 - t
    sigma_t = t
    x_t = alpha_t * x0 + sigma_t * noise
    return x_t, noise

def train(config, train_dataset):
    '''
    Run train
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compact_dataset = True

    dataset = DiffusionDataset(train_dataset, compact=compact_dataset)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = Pointmaze_MLP(input_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(config.n_train_iters):
        model.train()
        total_loss = 0

        for x0 in dataloader:
            x0 = x0.to(device)
            t = torch.rand(x0.shape[0], 1, device=device)
            x_t, noise = q_sample(x0, t)
            pred_noise = model(x_t, t)

            loss = loss_fn(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x0.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1:03d}/{config.n_train_iters} | Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "diffusion_model.pt")
    print("Model saved to diffusion_model.pt")
    return model
