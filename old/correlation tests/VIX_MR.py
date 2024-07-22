import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data from a CSV file
data = pd.read_csv("vxx_data.csv", parse_dates=['Date'], index_col='Date')

# Display the data
print(data.head())

# Step 2: Feature Engineering
# Calculate moving averages
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()

# Calculate RSI
def compute_RSI(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

data['RSI'] = compute_RSI(data)

# Calculate daily returns
data['Return'] = data['Close'].pct_change()

# Drop NaN values
data.dropna(inplace=True)

# Step 3: Strategy Development
def generate_signals(data):
    # Create a new column for signals
    data['Signal'] = 0
    # Buy signal
    data.loc[(data['SMA_20'] > data['SMA_50']) & (data['RSI'] < 30), 'Signal'] = 1
    # Sell signal
    data.loc[(data['SMA_20'] < data['SMA_50']) & (data['RSI'] > 70), 'Signal'] = -1
    return data

data = generate_signals(data)

# Step 4: Backtesting
# Initialize capital and positions
initial_capital = 100000
positions = pd.DataFrame(index=data.index).fillna(0.0)
portfolio = pd.DataFrame(index=data.index)

# Buy a number of shares
positions['VXX'] = data['Signal']

# Calculate the portfolio value
portfolio['positions'] = positions.multiply(data['Adj Close'], axis=0)
portfolio['cash'] = initial_capital - (positions.diff().multiply(data['Adj Close'], axis=0)).cumsum()
portfolio['total'] = portfolio['positions'] + portfolio['cash']

# Calculate portfolio returns
portfolio['returns'] = portfolio['total'].pct_change()

# Step 5: Performance Evaluation
# Calculate cumulative returns
cumulative_returns = (1 + portfolio['returns']).cumprod() - 1

# Calculate Sharpe ratio
sharpe_ratio = portfolio['returns'].mean() / portfolio['returns'].std() * np.sqrt(252)

# Display metrics
print(f"Cumulative Returns: {cumulative_returns.iloc[-1]:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(portfolio['total'], label='Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid()
plt.show()
