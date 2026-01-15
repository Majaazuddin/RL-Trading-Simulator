# 📈 Reinforcement Learning Trading Simulator

## 🔍 Problem Statement
Financial markets are highly volatile and unpredictable, making it difficult to design trading strategies that balance profit and risk. Traditional rule-based strategies often fail to adapt to changing market conditions.

The goal of this project is to build an **intelligent trading agent using Reinforcement Learning (RL)** that can learn trading decisions (Buy, Sell, Hold) directly from market data and **generalize well to unseen data**, while prioritizing **risk control and capital preservation**.

---

## 🧠 Approach
The project was developed in multiple stages, increasing complexity step by step:

1. **Data Collection & Processing**
   - Downloaded historical Bitcoin (BTC-USD) price data
   - Cleaned and preprocessed the data
   - Generated technical indicators such as SMA (20 & 50)
   - Normalized features for stable learning

2. **Trading Environment**
   - Designed a custom trading environment similar to OpenAI Gym
   - Defined state, action, and reward mechanisms
   - Simulated portfolio balance, positions, and rewards

3. **Reinforcement Learning Models**
   - Implemented a basic Q-Learning agent
   - Upgraded to a **Deep Q-Network (DQN)** using PyTorch
   - Used experience replay and target networks for stable training

4. **Training & Evaluation**
   - Trained the agent over multiple episodes
   - Evaluated performance on **unseen market data**
   - Compared results with Buy & Hold and Q-Learning strategies

---

## 🤖 Reinforcement Learning Explained (Simple Words)
Reinforcement Learning works by **learning from interaction**.

- The **Agent** (DQN) decides whether to Buy, Sell, or Hold
- The **Environment** simulates the market and portfolio
- The **State** includes price-based features (SMA, volume, etc.)
- The **Action** is Buy (1), Sell (2), or Hold (0)
- The **Reward** is the change in portfolio value

Over time, the agent learns which actions lead to better outcomes by maximizing long-term rewards rather than short-term gains.

---

## 📊 Results
The trained models were evaluated on unseen Bitcoin data to prevent overfitting.

### Strategy Comparison Summary:

| Strategy        | Final Value | Total Return | Max Drawdown | Sharpe Ratio |
|-----------------|------------|--------------|--------------|--------------|
| Buy & Hold      | 10250.00    | 2.5%         | -12.5%       | 0.85         |
| Q-Learning      | 10120.00    | 1.2%         | -4.8%        | 0.45         |
| DQN (Unseen)    | 10009.95    | 0.1%         | -0.93%       | 0.10         |

### Key Observations:
- Buy & Hold achieved higher returns but with high risk
- Q-Learning improved risk control
- **DQN demonstrated excellent capital preservation with minimal drawdown**
- The DQN agent generalized well to unseen data

---

## ⚠️ Limitations
- Returns are modest due to conservative behavior
- Transaction costs and slippage are simplified
- Model is trained on a single asset (Bitcoin)
- No sentiment or macroeconomic features are included
- DQN may underperform in strongly trending markets

---

## 🚀 Future Improvements
- Add transaction costs and slippage modeling
- Use multi-asset portfolios
- Incorporate LSTM or Transformer-based models
- Add sentiment and news-based features
- Improve reward shaping for better profitability

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy, Matplotlib
- PyTorch
- Reinforcement Learning (Q-Learning, DQN)

---

## 👨‍💻 Author
Mohammed Majaaz  
Aspiring Data Scientist / Machine Learning Engineer
