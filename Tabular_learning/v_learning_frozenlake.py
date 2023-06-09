'''
Description: None
Author: Bin Peng
Email: ustb_pengbin@163.com
Date: 2023-06-03 09:51:04
LastEditTime: 2023-06-03 11:07:58
'''
import gym
import collections
from torch.utils.tensorboard import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODES = 20

class Agent():
    def __init__(self) -> None:
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)


    def play_n_random_steps(self, count):
        for i in range(count):
            action = self.env.action_space.sample()
            newstate, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, newstate)] = reward
            self.transits[(self.state, action)][newstate] += 1
            self.state = self.env.reset() if is_done else newstate
    

    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            action_value += (count/total) * (reward + GAMMA * self.values[tgt_state])
        return action_value
    

    def select_actions(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action
    

    def play_episode(self, env:gym.Env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_actions(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward
    

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [self.calc_action_value(state, action) for action in range(self.env.action_space.n)]
            self.values[state] = max(state_values)
    

if __name__ == '__main__':
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-v-learning", log_dir='Tabular_learning/log/v_learning_frozenlake')

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no +=1 
        agent.play_n_random_steps(100)
        agent.value_iteration()
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward +=  agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated: %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.8:
            print("Sloved in %d iters" % iter_no)
            break
    writer.close()


