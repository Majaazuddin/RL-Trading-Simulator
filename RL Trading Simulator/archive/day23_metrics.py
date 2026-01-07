import numpy as np
import torch
import matplotlib.pyplot as plt

from env.trading_env import TradingEnvironment
from agents.dqn_agent import DQN

# =============================
# Load environment (UNSEEN DATA)
# =============================
env = TradingEnvironment("data/btc_unseen_features.csv")

STATE_SIZE = 4
ACTION_SIZE = 3
INITIAL_CAPITAL = 10000

# =============================
# Load trained DQN
# =============================
policy_net = DQN(STATE_SIZE, ACTION_SIZE)
policy_net.load_state_dict(torch.load("dqn_trading_model.pth"))
policy_net.eval()

print("✅ DQN model loaded")

# =============================
# Run backtest
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

portfolio_values = np.array(portfolio_values)

# =============================
# 1️⃣ Total Return
# =============================
final_value = portfolio_values[-1]
total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

# =============================
# 2️⃣ Maximum Drawdown
# =============================
running_max = np.maximum.accumulate(portfolio_values)
drawdowns = (portfolio_values - running_max) / running_max
max_drawdown = drawdowns.min()

# =============================
# 3️⃣ Sharpe Ratio
# =============================
returns = np.diff(portfolio_values) / portfolio_values[:-1]

if returns.std() == 0:
    sharpe_ratio = 0
else:
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)

# =============================
# Print metrics
# =============================
print("\n📊 PERFORMANCE METRICS (UNSEEN DATA)")
print("----------------------------------")
print(f"Initial Capital : {INITIAL_CAPITAL:.2f}")
print(f"Final Value     : {final_value:.2f}")
print(f"Total Return    : {total_return * 100:.2f}%")
print(f"Max Drawdown    : {max_drawdown * 100:.2f}%")
print(f"Sharpe Ratio    : {sharpe_ratio:.2f}")

# =============================
# Plot portfolio
# =============================
plt.figure()
plt.plot(portfolio_values)
plt.xlabel("Time Step")
plt.ylabel("Portfolio Value")
plt.title("DQN Backtest + Risk Metrics (Unseen Data)")
plt.show()
