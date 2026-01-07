import yfinance as yf
import os

# -----------------------------
# Create data folder if not exists
# -----------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# Download Bitcoin data
# -----------------------------
ticker = "BTC-USD"

print("Downloading Bitcoin data...")

btc_data = yf.download(
    ticker,
    start="2019-01-01",
    progress=True
)

# -----------------------------
# Save data as CSV
# -----------------------------
file_path = os.path.join(DATA_DIR, "btc_usd.csv")
btc_data.to_csv(file_path)

print("Data saved successfully at:", file_path)
print(btc_data.head())
