import numpy as np
import random
from collections import deque

from qnetwork import QNetwork

import torch
import torch.optim as optim


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:

    def __init__(self, state_size, action_size, lr=5e-4, gamma=0.99, tau=1e-3, update_freq=4, buffer_size=int(1e5), batch_size=64):
        """
        Interacts with and learns from the environment.
        :param state_size: (int)
        :param action_size: (int)
        :param lr: (float)
        :param gamma: (float) discount factor
        :param tau: (float) soft update rate of target parameters
        :param update_freq: (int) update local & target network every n steps
        :param buffer_size: (int)
        :param batch_size: (int) how many samples to use when doing single update
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.update_freq = update_freq
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # Q-Network
        self.qnetwork_local = QNetwork(self.state_size, self.action_size).to(DEVICE)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size).to(DEVICE)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size)

        # Initialize time step
        self.t_step = 0

    def step(self, e):
        """
        process from one step of interaction experience with environment
        :param e: experience dictionary with keys {state, action, reward, next_state, done}
        """
        self.t_step += 1
        # Save experience in replay memory
        self.memory.add(e)
        # Learn every update_freq steps
        if self.t_step % self.update_freq == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.):
        """
        produce action from state using local qnetwork and epsilon-greedy policy
        :param state: (np.array)
        :param eps: (float) epsilon for epsilon-greedy
        :return: action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        # epsilon greedy
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, e):
        """
        Update value parameters using given batch of experience tuples.
        :param e: experience dictionary with keys {state, action, reward, next_state, done}
        """
        # local network gradient descent (double dqn)
        with torch.no_grad():
            next_action = self.qnetwork_local(e['next_state']).detach().max(-1, keepdim=True)[1]
            next_q = self.qnetwork_target(e['next_state']).detach().gather(1, next_action).squeeze(1)
            # next_q = self.qnetwork_target(e['next_state']).detach().max(-1, keepdim=True)[0].squeeze(1)
            target = e['reward'] + self.gamma * next_q * (1 - e['done'])
        pred = self.qnetwork_local(e['state']).gather(1, e['action'].unsqueeze(1)).squeeze(1)
        loss = ((target - pred) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # update target network towards local network
        self.soft_update()

    def soft_update(self):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


class ReplayBuffer:
    """"""

    def __init__(self, action_size, buffer_size, batch_size):
        """
        Fixed-size buffer to store experience tuples.
        :param action_size: (int)
        :param buffer_size: (int)
        :param batch_size: (int)
        """
        self.action_size = action_size
        self.memory_keys = {'state', 'action', 'reward', 'next_state', 'done'}
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, experience_dict):
        """
        Add a new experience to memory.
        :param experience_dict: experience dictionary with keys {state, action, reward, next_state, done}
        """
        assert self.memory_keys == set(experience_dict.keys())
        self.memory.append(experience_dict)

    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        out = {k: torch.from_numpy(np.stack([e[k] for e in experiences], axis=0)).to(device=DEVICE, dtype=torch.int64 if k in ['action'] else torch.float32) for k in self.memory_keys}
        return out

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.memory)
