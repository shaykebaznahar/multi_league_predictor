import pandas as pd

print("🔄 Loading data...")

# Load training data (2005-2019)
train_df = pd.read_csv('data/train/all_leagues_2005_2019.csv')

# Load testing data (2020-2024)
test_df = pd.read_csv('data/test/all_leagues_2020_2024.csv')

print(f"✅ Train data: {len(train_df)} rows")
print(f"✅ Test data: {len(test_df)} rows")

# Check columns
print(f"\nColumns in train: {list(train_df.columns)}")

# Combine both
all_data = pd.concat([train_df, test_df], ignore_index=True)

print(f"\n✅ Combined: {len(all_data)} rows")

# Check leagues
if 'Div' in all_data.columns:
    print(f"\nLeagues found: {sorted(all_data['Div'].unique())}")
    print(f"Number of leagues: {all_data['Div'].nunique()}")

# Extract year if Date column exists
if 'Date' in all_data.columns:
    all_data['Date'] = pd.to_datetime(all_data['Date'], format='%d/%m/%Y', errors='coerce')
    all_data['Year'] = all_data['Date'].dt.year
    print(f"\nYear range: {all_data['Year'].min()} to {all_data['Year'].max()}")

# Save combined
all_data.to_csv('data/combined/all_leagues_2005_2024.csv', index=False)

print("\n✅ Done! Saved to: data/combined/all_leagues_2005_2024.csv")