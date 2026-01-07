import pandas as pd
import matplotlib.pyplot as plt
import torch

from env.trading_env import TradingEnvironment
from agents.dqn_agent import DQN

# =============================
# Load data & model
# =============================
DATA_PATH = "data/btc_unseen_features.csv"
MODEL_PATH = "dqn_trading_model.pth"

env = TradingEnvironment(DATA_PATH)

STATE_SIZE = 4
ACTION_SIZE = 3

model = DQN(STATE_SIZE, ACTION_SIZE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# =============================
# Containers for plotting
# =============================
prices = []
portfolio_values = []
buy_points = []
sell_points = []

# =============================
# Run agent
# =============================
state = env.reset()
done = False
step = 0

while not done:
    # ✅ FIX: get price correctly
    price = env.data.iloc[env.current_step]["Close"]
    prices.append(price)
    portfolio_values.append(env.portfolio_value)

    state_tensor = torch.tensor(state, dtype=torch.float32)

    with torch.no_grad():
        action = torch.argmax(model(state_tensor)).item()

    if action == 1:  # BUY
        buy_points.append((step, price))
    elif action == 2:  # SELL
        sell_points.append((step, price))

    state, reward, done = env.step(action)
    step += 1

# =============================
# Convert to DataFrame
# =============================
df = pd.DataFrame({
    "Price": prices,
    "Portfolio": portfolio_values
})

# =============================
# Plot 1: Price + Buy/Sell
# =============================
plt.figure(figsize=(12, 5))
plt.plot(df["Price"], label="BTC Price", color="blue")

if buy_points:
    x, y = zip(*buy_points)
    plt.scatter(x, y, marker="^", color="green", label="Buy", s=80)

if sell_points:
    x, y = zip(*sell_points)
    plt.scatter(x, y, marker="v", color="red", label="Sell", s=80)

plt.title("BTC Price with Buy / Sell Decisions")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

# =============================
# Plot 2: Equity Curve
# =============================
plt.figure(figsize=(12, 5))
plt.plot(df["Portfolio"], label="Portfolio Value", color="purple")
plt.title("Equity Curve (Portfolio Value Over Time)")
plt.xlabel("Time Step")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid()
plt.show()
