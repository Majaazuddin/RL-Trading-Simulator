import pandas as pd

# =============================
# 1️⃣ Load cleaned Bitcoin data
# =============================
file_path = "data/btc_usd_cleaned.csv"
df = pd.read_csv(file_path)

print("✅ Cleaned data loaded")
print(df.head())

# =============================
# 2️⃣ Calculate Simple Moving Averages
# =============================

# SMA 20: average of last 20 days
df["SMA_20"] = df["Close"].rolling(window=20).mean()

# SMA 50: average of last 50 days
df["SMA_50"] = df["Close"].rolling(window=50).mean()

print("\n✅ SMA columns added")
print(df[["Close", "SMA_20", "SMA_50"]].head(25))

# =============================
# 3️⃣ Remove rows with NaN (caused by SMA)
# =============================
df.dropna(inplace=True)

print("\n✅ Rows with NaN removed")
print(df.head())

# =============================
# 4️⃣ Normalize data (VERY IMPORTANT FOR RL)
# =============================
columns_to_scale = ["Close", "SMA_20", "SMA_50", "Volume"]

for col in columns_to_scale:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

print("\n✅ Data normalized (scaled)")
print(df[columns_to_scale].head())

# =============================
# 5️⃣ Save final feature dataset
# =============================
final_file_path = "data/btc_features.csv"
df.to_csv(final_file_path, index=False)

print("\n🎉 Feature-engineered data saved at:", final_file_path)
