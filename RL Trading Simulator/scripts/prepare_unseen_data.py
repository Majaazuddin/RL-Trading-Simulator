import pandas as pd
import numpy as np

print("🔄 Preparing unseen BTC data...")

# =============================
# Load raw unseen data
# =============================
df = pd.read_csv("data/btc_unseen.csv")

# Fix Date column if needed
if "Date" not in df.columns:
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Date"}, inplace=True)

# Keep only required columns
df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

# Convert to numeric
for col in ["Open", "High", "Low", "Close", "Volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =============================
# Feature engineering (SAME AS TRAINING)
# =============================
df["SMA_20"] = df["Close"].rolling(20).mean()
df["SMA_50"] = df["Close"].rolling(50).mean()

# Remove NaN rows
df.dropna(inplace=True)

# =============================
# Normalization
# =============================
for col in ["Close", "SMA_20", "SMA_50", "Volume"]:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

# =============================
# Save FINAL unseen feature file
# =============================
df.to_csv("data/btc_unseen_features.csv", index=False)

print("✅ File created: data/btc_unseen_features.csv")
print(df.head())
