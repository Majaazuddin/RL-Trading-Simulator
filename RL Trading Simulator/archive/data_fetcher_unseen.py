import yfinance as yf
import os

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

print("⬇️ Downloading UNSEEN Bitcoin data...")

btc = yf.download(
    "BTC-USD",
    start="2024-01-01",
    progress=False
)

btc.to_csv("data/btc_unseen.csv")

print("✅ Unseen BTC data saved to data/btc_unseen.csv")
print(btc.head())
