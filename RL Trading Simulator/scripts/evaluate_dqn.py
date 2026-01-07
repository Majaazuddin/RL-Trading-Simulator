import numpy as np
import torch
import matplotlib.pyplot as plt

from env.trading_env import TradingEnvironment
from agents.dqn_agent import DQN

# =============================
# 1️⃣ Load environment
# =============================
env = TradingEnvironment("data/btc_features.csv")

STATE_SIZE = 4
ACTION_SIZE = 3

# =============================
# 2️⃣ Load trained DQN
# =============================
policy_net = DQN(STATE_SIZE, ACTION_SIZE)
policy_net.load_state_dict(torch.load("dqn_trading_model.pth"))
policy_net.eval()

# =============================
# 3️⃣ Run evaluation
# =============================
state = env.reset()
state = torch.tensor(
    np.array(state, dtype=np.float32),
    dtype=torch.float32
)

done = False
portfolio_values = []

while not done:
    with torch.no_grad():
        action = torch.argmax(policy_net(state)).item()

    next_state, reward, done = env.step(action)

    state = torch.tensor(
        np.array(next_state, dtype=np.float32),
        dtype=torch.float32
    )

    portfolio_values.append(env.portfolio_value)

# =============================
# 4️⃣ Plot portfolio value
# =============================
plt.figure()
plt.plot(portfolio_values)
plt.xlabel("Time Step")
plt.ylabel("Portfolio Value")
plt.title("DQN Portfolio Value Over Time")
plt.show()

print(f"💰 Final Portfolio Value: {env.portfolio_value:.2f}")
