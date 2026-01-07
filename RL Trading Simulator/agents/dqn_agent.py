import torch
import torch.nn as nn
import torch.optim as optim

# =============================
# DQN Neural Network
# =============================
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        # -----------------------------
        # Fully connected layers
        # -----------------------------
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """
        Forward pass:
        state -> Q-values
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
