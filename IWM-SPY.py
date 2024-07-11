import pandas as pd

# Load data from CSV files
stock1 = pd.read_csv('IWM.csv', parse_dates=['Date'], index_col='Date')
stock2 = pd.read_csv('SPY.csv', parse_dates=['Date'], index_col='Date')

# Calculate daily returns
stock1['Daily Return'] = stock1['Adj Close'].pct_change()
stock2['Daily Return'] = stock2['Adj Close'].pct_change()

# Merge the dataframes on the date
merged_data = pd.merge(stock1['Daily Return'], stock2['Daily Return'], left_index=True, right_index=True, suffixes=('_stock1', '_stock2'))

# Drop missing values
merged_data = merged_data.dropna()

# Calculate the Pearson correlation coefficient
correlation = merged_data.corr().iloc[0, 1]

print(f"The correlation between the two stocks is {correlation:.2f}")
