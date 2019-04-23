from dqn import DQN
import torch
import gym
import gym_interf


def make_env(seed=None):
    env = gym.make('interf-v1')
    if seed is not None:
        env.seed(seed)
    return env


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DQN(make_env, device, save_dir='mymodel', buffer_size=1, total_steps=1)
#model.learn()
#model.save()
model.load()