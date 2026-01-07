import random
from collections import deque

# =============================
# Experience Replay Buffer
# =============================
class ReplayBuffer:
    def __init__(self, capacity):
        """
        capacity: maximum number of experiences to store
        """
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store one experience in memory
        """
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Return current number of stored experiences
        """
        return len(self.memory)
