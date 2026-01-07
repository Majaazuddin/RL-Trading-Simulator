import pandas as pd

# =============================
# 1️⃣ Load raw Bitcoin data
# =============================
file_path = "data/btc_usd.csv"
df = pd.read_csv(file_path)

print("✅ Raw data loaded")
print(df.head())

# =============================
# 2️⃣ Remove junk rows (Ticker & Date rows)
# =============================
# Keep only rows where 'Price' looks like a date (YYYY-MM-DD)
df = df[df["Price"].str.match(r"\d{4}-\d{2}-\d{2}", na=False)]

print("\n✅ Junk rows removed")
print(df.head())

# =============================
# 3️⃣ Rename Price → Date
# =============================
df.rename(columns={"Price": "Date"}, inplace=True)

# =============================
# 4️⃣ Convert Date column
# =============================
df["Date"] = pd.to_datetime(df["Date"])

# =============================
# 5️⃣ Convert price columns to numeric
# =============================
numeric_cols = ["Open", "High", "Low", "Close", "Volume"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =============================
# 6️⃣ Remove missing values
# =============================
df.dropna(inplace=True)

# =============================
# 7️⃣ Set Date as index
# =============================
df.set_index("Date", inplace=True)

# =============================
# 8️⃣ Keep only required columns
# =============================
df = df[["Open", "High", "Low", "Close", "Volume"]]

print("\n✅ Cleaned data preview:")
print(df.head())

print("\n✅ Data types after cleaning:")
print(df.dtypes)

# =============================
# 9️⃣ Save cleaned data
# =============================
clean_file_path = "data/btc_usd_cleaned.csv"
df.to_csv(clean_file_path)

print("\n🎉 Cleaned data saved at:", clean_file_path)
