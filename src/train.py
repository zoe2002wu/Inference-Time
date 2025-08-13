import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model.pointmaze_mlp import Pointmaze_MLP
import numpy as np
import matplotlib.pyplot as plt

def parse_data(data):
    obs = data['observations']
    valids = data['valids']
    valids_indices = np.where(valids == 0.)[0]
    chunk_size = valids_indices[0]
    split_dataset = [torch.tensor(obs[i-chunk_size:i-1]) for i in valids_indices]
    return torch.stack(split_dataset, dim=0)


def q_sample(x0, t):
    '''
    Forward Process
    '''
    noise = torch.randn_like(x0)
    alpha_t = (1 - t).view(-1, 1, 1)  # shape: [128, 1, 1]
    sigma_t = t.view(-1, 1, 1)        # shape: [128, 1, 1]
    x_t = alpha_t * x0 + sigma_t * noise
    return x_t, noise

def visualize_on_ogbench_env(env, traj, task_id=1, title="Sampled Trajectories", every_k=5, particle_idx: int = 0, delay: float = 0.05):

    for pos in traj:
        qpos = pos  # [2]
        qvel = np.zeros_like(qpos)  # [2] or whatever matches the env

        env.unwrapped.set_state(qpos, qvel)

        frame = env.render()
        if isinstance(frame, np.ndarray):
            plt.imshow(frame)
            plt.axis('off')
            plt.pause(delay)
            plt.clf()
        else:
            time.sleep(delay)

    env.close()

def train(config, train_dataset, env):
    '''
    Run train
    '''
    dataset = parse_data(train_dataset)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    device = "cpu"

    ob, info = env.reset(
        options=dict(
            task_id=config.task_id,  # Set the evaluation task. Each environment provides five
                                # evaluation goals, and `task_id` must be in [1, 5].
            render_goal=True,  # Set to `True` to get a rendered goal image (optional).
        )
    )

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
