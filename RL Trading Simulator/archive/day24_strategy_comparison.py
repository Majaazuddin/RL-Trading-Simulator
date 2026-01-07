import pandas as pd

# =============================
# Strategy Results (MANUAL INPUT)
# =============================
# Fill these values from your outputs

data = {
    "Strategy": [
        "Buy & Hold",
        "Q-Learning",
        "DQN (Unseen Data)"
    ],
    "Final Value": [
        10250.00,   # example – replace with your Buy & Hold result
        10120.00,   # example – replace with Q-learning result
        10009.95    # from your Day 23 output
    ],
    "Total Return (%)": [
        2.50,
        1.20,
        0.10
    ],
    "Max Drawdown (%)": [
        -12.50,
        -4.80,
        -0.93
    ],
    "Sharpe Ratio": [
        0.85,
        0.45,
        0.10
    ]
}

# =============================
# Create comparison table
# =============================
df = pd.DataFrame(data)

print("\n📊 STRATEGY COMPARISON TABLE")
print("----------------------------------")
print(df)

# =============================
# Save to CSV (for report)
# =============================
df.to_csv("strategy_comparison.csv", index=False)
print("\n✅ Strategy comparison saved as strategy_comparison.csv")
