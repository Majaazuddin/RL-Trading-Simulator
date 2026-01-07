import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.trading_env import TradingEnvironment
from agents.dqn_agent import DQN
from agents.replay_buffer import ReplayBuffer

# =============================
# 1️⃣ Hyperparameters
# =============================
STATE_SIZE = 4
ACTION_SIZE = 3

GAMMA = 0.95
LEARNING_RATE = 0.0005
BATCH_SIZE = 128
MEMORY_SIZE = 10000

EPISODES = 50           # keep small for now
TARGET_UPDATE = 10      # update target net every 10 episodes

EPSILON = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995

# =============================
# 2️⃣ Environment & Networks
# =============================
env = TradingEnvironment("data/btc_features.csv")

policy_net = DQN(STATE_SIZE, ACTION_SIZE)
target_net = DQN(STATE_SIZE, ACTION_SIZE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

memory = ReplayBuffer(MEMORY_SIZE)

# =============================
# 3️⃣ Training Loop
# =============================
for episode in range(EPISODES):

    # 🔹 RESET ENVIRONMENT
    state = env.reset()

    # ✅ FIX: force numeric float32 state
    state = torch.tensor(
        np.array(state, dtype=np.float32),
        dtype=torch.float32
    )

    total_reward = 0
    done = False

    while not done:
        # -----------------------------
        # Action selection (epsilon-greedy)
        # -----------------------------
        if random.random() < EPSILON:
            action = random.randrange(ACTION_SIZE)
        else:
            with torch.no_grad():
                action = torch.argmax(policy_net(state)).item()

        # -----------------------------
        # Environment step
        # -----------------------------
        next_state, reward, done = env.step(action)

        # ✅ FIX: force numeric float32 next_state
        next_state_tensor = torch.tensor(
            np.array(next_state, dtype=np.float32),
            dtype=torch.float32
        )

        # Store experience
        memory.push(state, action, reward, next_state_tensor, done)

        state = next_state_tensor
        total_reward += reward

        # -----------------------------
        # Train only if memory is ready
        # -----------------------------
        if len(memory) < BATCH_SIZE:
            continue

        # -----------------------------
        # Sample batch
        # -----------------------------
        batch = memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        # -----------------------------
        # Compute Q-values
        # -----------------------------
        current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            max_next_q = target_net(next_states).max(1)[0]
            target_q = rewards + GAMMA * max_next_q * (1 - dones)

        # -----------------------------
        # Backpropagation
        # -----------------------------
        loss = loss_fn(current_q, target_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # -----------------------------
    # Epsilon decay
    # -----------------------------
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

    # -----------------------------
    # Target network update
    # -----------------------------
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(
        f"Episode {episode + 1}/{EPISODES} | "
        f"Total Reward: {total_reward:.2f} | "
        f"Epsilon: {EPSILON:.3f}"
    )

print("\n🎉 DQN Training Finished!")
# =============================
# Save trained DQN model
# =============================
torch.save(policy_net.state_dict(), "dqn_trading_model.pth")
print("✅ Trained DQN model saved as dqn_trading_model.pth")
# =============================
# Save trained DQN model
# =============================
torch.save(policy_net.state_dict(), "dqn_trading_model.pth")

print("✅ DQN model saved as dqn_trading_model.pth")
