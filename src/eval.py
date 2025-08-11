import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import ogbench
from model.pointmaze_mlp import Pointmaze_MLP
import gym


# -------------------------
# DDIM Reverse Sampling
# -------------------------
@torch.no_grad()
def reverse(x_t: torch.Tensor, model: nn.Module, T: int, device: str) -> torch.Tensor:
    """Reverse diffusion: Generate trajectory from noise x_T."""
    model.eval()
    trajectory = [x_t]

    for t_val in reversed(range(1, T + 1)):
        t = (1 - 1e-3) * t_val / T
        t_next = (1 - 1e-3) * (t_val - 1) / T

        N = x_t.shape[0]
        t_tensor = torch.full((N, 1), t, dtype=torch.float32, device=device)

        alpha_t = 1 - t
        alpha_next = 1 - t_next
        sigma_t = t
        sigma_next = t_next

        eps_theta = model(x_t, t_tensor)
        x_t = (alpha_next / alpha_t) * (x_t - sigma_t * eps_theta) + sigma_next * eps_theta

        trajectory.append(x_t)

    return torch.stack(trajectory, dim=0)  # Shape: [T+1, N, D]


# -------------------------
# Search-Guided Sampling
# -------------------------
def verifier(x: torch.Tensor) -> torch.Tensor:
    """
    Task-specific verifier score. Higher = better.
    Replace this with goal-conditioned or learned verifier.
    """
    return torch.ones(x.shape[0], device=x.device)


def langevin_search(
    x_t: torch.Tensor,
    t: torch.Tensor,
    n_children: list[int],
    model: nn.Module,
    steps: int = 5,
    dt: float = 0.1
) -> torch.Tensor:
    """
    Apply Langevin dynamics with model + verifier gradients.
    """
    x_children = []

    for i, x in enumerate(x_t):
        repeated = x.repeat(n_children[i], 1)
        x_children.append(repeated)

    x_children = torch.cat(x_children, dim=0).detach().requires_grad_(True)

    for _ in range(steps):
        score = model(x_children, t)
        verifier_score = verifier(x_children).sum()

        (-verifier_score).backward(retain_graph=True)
        grad_verifier = x_children.grad.clone()
        x_children.grad.zero_()

        total_score = score + grad_verifier
        noise = torch.randn_like(x_children)
        x_children = (
            x_children - dt * total_score + torch.sqrt(torch.tensor(2 * dt)) * noise
        )
        x_children = x_children.detach().requires_grad_(True)

    return x_children.detach()


@torch.no_grad()
def search_guided_sample(
    x_T: torch.Tensor,
    model: nn.Module,
    T: int,
    device: str,
    n_particles: int,
    langevin_steps: int = 5
) -> torch.Tensor:
    """
    Guided sampling with BFS-style resampling and verifier.
    """
    particles = x_T
    dt = 1/T
    trajectories = [particles]

    for t_val in reversed(range(1, T + 1)):
        t = (1 - 1e-3) * t_val / T
        scores = verifier(particles)
        n_children = torch.round(scores * n_particles / scores.sum()).to(torch.int).tolist()
        t_tensor = torch.full((sum(n_children), 1), t, dtype=torch.float32, device=device)

        particles = langevin_search(particles, t_tensor, n_children, model, steps=langevin_steps, dt=dt)
        trajectories.append(particles)

    # Stack to shape [T+1, N, D] (may be uneven if resampling isn't perfect)
    min_len = min(x.shape[0] for x in trajectories)
    clipped = [x[:min_len] for x in trajectories]
    return torch.stack(clipped, dim=0)


# -------------------------
# Visualization
# -------------------------
def visualize_on_ogbench_env(
    env,
    trajectory: torch.Tensor,
    task_id: int = 1,
    goal: np.ndarray = None,
    title: str = "Sampled Trajectories",
    every_k: int = 5
):
    """
    Visualize trajectory on environment rendering.
    """


    # Evaluate the agent.
    ob, info = env.reset()  # Reset the environment.

    done = False
    while not done:
        action = env.action_space.sample()  # Replace this with your agent's action.
        ob, reward, terminated, truncated, info = env.step(action)  # Gymnasium-style step.
        # If the agent achieves the task, `terminated` will be `True`. If the episode length
        # exceeds the maximum length without achieving the task, `truncated` will be `True`.
        done = terminated or truncated
        env.render()  # Render the current frame (optional).

    env.close()

    # _, info = env.reset(options=dict(task_id=task_id, render_goal=True))
    # render = env.render()

    # traj_np = trajectory.cpu().numpy()
    # T_plus_1, N, D = traj_np.shape
    # assert D >= 2, "Trajectory must have at least 2D positions"

    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.imshow(render, origin='lower')
    # ax.set_title(title)

    # bounds = env.unwrapped.boundaries
    # low, high = np.array(bounds[0]), np.array(bounds[1])

    # def to_pixel(pos):
    #     norm = (pos - low) / (high - low)
    #     h, w, *_ = render.shape
    #     return np.stack([norm[:, 0] * w, norm[:, 1] * h], axis=-1)

    # for i in range(0, N, every_k):
    #     traj = traj_np[:, i, :2]
    #     pix_traj = to_pixel(traj)
    #     ax.plot(pix_traj[:, 0], pix_traj[:, 1], alpha=0.8, linewidth=1.5)
    #     ax.scatter(pix_traj[0, 0], pix_traj[0, 1], color='red', s=20, label='Start' if i == 0 else "")
    #     ax.scatter(pix_traj[-1, 0], pix_traj[-1, 1], color='green', s=20, label='End' if i == 0 else "")

    # if goal is not None:
    #     goal_pix = to_pixel(goal[None, :])[0]
    #     ax.scatter(goal_pix[0], goal_pix[1], color='blue', s=30, marker='*', label='Goal')

    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_aspect("equal")
    # ax.legend()
    # plt.tight_layout()
    # plt.show()


# -------------------------
# Main Entry
# -------------------------
def sample(config):
    # device =  "cpu"

    # model = Pointmaze_MLP(input_dim=2).to(device)
    # model.load_state_dict(torch.load(config.model_path, map_location=device))

    # env, _, _ = ogbench.make_env_and_datasets(config.dataset_name)
    # _, info = env.reset(options=dict(task_id=config.task_id))
    # goal = info['goal'][:2]

    # x_T = torch.randn(config.n_particles, 2, device=device)

    # if config.eval_mode == "ddim":
    #     trajectory = reverse(x_T, model=model, T=config.n_discrete_steps, device=device)
    # elif config.eval_mode == "bfs":
    #     trajectory = search_guided_sample(
    #         x_T,
    #         model=model,
    #         T=config.n_discrete_steps,
    #         device=device,
    #         n_particles=config.n_particles,
    #         langevin_steps=config.langevin_steps
    #     )
    # else:
    #     raise ValueError(f"Unsupported eval_mode: {config.eval_mode}")

    # visualize_on_ogbench_env(env, trajectory, task_id=config.task_id, goal=goal)

    env = gym.make("CartPole-v1", render_mode="human")
    ob, info = env.reset()

    for _ in range(100):
        action = env.action_space.sample()
        ob, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            break

    env.close()
