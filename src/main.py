import argparse as p
from train import train
import ogbench

def run_main(config):
    dataset_name = 'humanoidmaze-large-navigate-v0'
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name)

    if config.mode == 'train':
        train(config, train_dataset)
    elif config.mode == 'eval':
        run_evaluation(config)
    elif config.mode == 'continue':
        print("Continue training not yet implemented.")

if __name__ == '__main__':
    parser = p.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval', 'continue'], required=True)
    parser.add_argument('--dataset_name', type=str, default='pointmaze-medium-navigate-v0')
    parser.add_argument('--task_id', type=int, default=1)

    #train
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_train_iters', type=int, default=100)

    #test
    parser.add_argument('--eval_mode', choices=['ddim', 'bfs', 'dfs'], required=True)
    parser.add_argument('--n_discrete_steps', type=int, default=100)
    parser.add_argument('--model_path', type=str, default='diffusion_model.pt')

    #test bfs
    parser.add_argument('--n_particles', type=int, default=256)
    parser.add_argument('--langevin_steps', type=int, default=5)

    config = parser.parse_args()
    run_main(config)
