import numpy as np
import torch
import matplotlib.pyplot as plt

from env.trading_env import TradingEnvironment
from agents.dqn_agent import DQN

# =============================
# Load unseen environment
# =============================
env = TradingEnvironment("data/btc_unseen_features.csv")

STATE_SIZE = 4
ACTION_SIZE = 3

# =============================
# Load trained model
# =============================
policy_net = DQN(STATE_SIZE, ACTION_SIZE)
policy_net.load_state_dict(torch.load("dqn_trading_model.pth"))
policy_net.eval()

print("✅ Loaded trained DQN model")

# =============================
# Run backtest (NO training)
# =============================
state = env.reset()
state = torch.tensor(np.array(state, dtype=np.float32))

done = False
portfolio_values = []

while not done:
    with torch.no_grad():
        action = torch.argmax(policy_net(state)).item()

    next_state, reward, done = env.step(action)
    state = torch.tensor(np.array(next_state, dtype=np.float32))
    portfolio_values.append(env.portfolio_value)

# =============================
# Plot results
# =============================
plt.figure()
plt.plot(portfolio_values)
plt.xlabel("Time Step")
plt.ylabel("Portfolio Value")
plt.title("DQN Backtest on Unseen BTC Data")
plt.show()

print(f"💰 Final Portfolio Value (Unseen Data): {env.portfolio_value:.2f}")
