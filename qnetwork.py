import torch
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, hidden_size=64):
        """
        Double DQN
        :param state_size: (int)
        :param action_size: (int)
        :param hidden_size: (int)
        """
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU()
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        """
        Build a network that maps state -> action values.
        """
        x = self.fc(state)
        value = self.value_head(x)
        adv = self.adv_head(x)
        q = value + adv - torch.mean(adv, dim=1, keepdim=True)
        return q
