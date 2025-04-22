import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Load data
df = pd.read_csv("dog_rates_tweets.csv")

# Convert created_at to datetime
df['date'] = pd.to_datetime(df['created_at'], errors='coerce')

# Extract rating using regex
def extract_rating(text):
    match = re.search(r'(\d+(\.\d+)?)/10', text)
    return float(match.group(1)) if match else None

df['rating'] = df['text'].apply(extract_rating)

# Drop NaNs and filter ratings > 20
df_clean = df.dropna(subset=['rating']).copy()
df_clean = df_clean[df_clean['rating'] <= 20]

# Convert date to numeric timestamp for regression
df_clean['date_num'] = df_clean['date'].astype(np.int64) // 10**9

# Sort by date
df_clean = df_clean.sort_values(by='date')

# Plot 1: Scatter with regression line
x = df_clean['date_num']
y = df_clean['rating']
coeffs = np.polyfit(x, y, 1)
trend = np.poly1d(coeffs)

plt.figure(figsize=(12, 6))
plt.scatter(df_clean['date'], y, alpha=0.3, label='Ratings')
plt.plot(df_clean['date'], trend(x), color='red', label='Trend line')
plt.title("Pup Inflation Over Time")
plt.xlabel("Date")
plt.ylabel("Rating (out of 10)")
plt.legend()
plt.tight_layout()
plt.savefig("pup_inflation_scatter.png")
plt.close()

# Plot 2: Boxplot by year
df_clean['year'] = df_clean['date'].dt.year
years = sorted(df_clean['year'].unique())
data_by_year = [df_clean[df_clean['year'] == year]['rating'] for year in years]

plt.figure(figsize=(12, 6))
plt.boxplot(data_by_year, labels=years, showfliers=True)
plt.axhline(10, color='red', linestyle='--', label='10/10 line')
plt.title("Distribution of Dog Ratings by Year")
plt.xlabel("Year")
plt.ylabel("Rating (out of 10)")
plt.legend()
plt.tight_layout()
plt.savefig("pup_inflation_boxplot.png")
plt.close()
