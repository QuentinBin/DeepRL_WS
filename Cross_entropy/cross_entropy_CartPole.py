'''
Description: None
Author: Bin Peng
Email: ustb_pengbin@163.com
Date: 2023-05-30 23:16:11
LastEditTime: 2023-05-31 20:04:04
'''
#!/usr/bin/env python3
import gym
from collections import namedtuple
import numpy as np
from torch.utils.tensorboard import SummaryWriter


import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTLIE = 70


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
    rewards = list(map(lambda s:s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_acts = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step:step.observation, example.steps))
        train_acts.extend(map(lambda step:step.action, example.steps))

    train_acts_v = torch.LongTensor(train_acts)
    train_obs_v = torch.FloatTensor(train_obs)
    return train_obs_v, train_acts_v, reward_bound, reward_mean


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole", log_dir='Cross_entropy/log/cartpole')

    for iter_no, batch in enumerate(iterate_bathches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTLIE)
        optimizer.zero_grad()
        action_score_v = net(obs_v)
        loss_v = objective(action_score_v, acts_v)
        loss_v.backward()
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" %(iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)

        if reward_m > 199:
            print("Solved!")
            break
    writer.close()
