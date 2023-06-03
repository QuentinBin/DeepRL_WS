'''
Description: None
Author: Bin Peng
Email: ustb_pengbin@163.com
Date: 2023-05-31 19:02:42
LastEditTime: 2023-05-31 21:24:24
'''
#!/usr/bin/env python3
import gym
import gym.spaces
from collections import namedtuple
from gym.core import Env
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTLIE = 30
GAMMA = 0.9

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions) -> None:
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env: Env) -> None:
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n,), dtype=np.float32)

    def observation(self, observation):
        # print(self.observation_space.low)
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res


def iterate_bathches(env:gym.Env, net:Net, batch_size:int):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)

    while True:
        obs_v = torch.FloatTensor(obs)
        act_probs_v = sm(net(obs_v.unsqueeze(0)))
        act_probs = act_probs_v.data.numpy()[0]

        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)

        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))

        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s:s.reward*(GAMMA**len(s.steps)), batch))
    reward_bound = np.percentile(rewards, percentile)

    train_obs = []
    train_acts = []
    elite_batch = []
    for example, reward in zip(batch, rewards):
        if reward > reward_bound:
            train_obs.extend(map(lambda step:step.observation, example.steps))
            train_acts.extend(map(lambda step:step.action, example.steps))
            elite_batch.append(example)

    train_acts_v = torch.LongTensor(train_acts)
    train_obs_v = torch.FloatTensor(train_obs)
    return train_obs_v, train_acts_v, reward_bound, elite_batch


if __name__ == '__main__':
    env = DiscreteOneHotWrapper(gym.make("FrozenLake-v0"))
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-frozenLake", log_dir='Cross_entropy/log/frozenlake')

    full_batch = []
    for iter_no, batch in enumerate(iterate_bathches(env, net, BATCH_SIZE)):
        reward_m = float(np.mean(list(map(lambda s: s.reward, batch))))
        obs_v, acts_v, reward_b, full_batch = filter_batch(full_batch+batch, PERCENTLIE)
        if not full_batch:
            continue
        full_batch = full_batch[-500:]
        optimizer.zero_grad()
        action_score_v = net(obs_v)
        loss_v = objective(action_score_v, acts_v)
        loss_v.backward()
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" %(iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)

        if reward_m > 0.8:
            print("Solved!")
            break
    writer.close()

