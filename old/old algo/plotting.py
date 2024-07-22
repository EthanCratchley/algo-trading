import pandas as pd
import matplotlib.pyplot as plt

# Load the news sentiment data
news_sentiment = pd.read_csv('news_sentiment_updated.csv')

# Load the SPY data
spy_data = pd.read_csv('SPY (1).csv')

# Convert the date columns to datetime format
news_sentiment['date'] = pd.to_datetime(news_sentiment['date'])
spy_data['Date'] = pd.to_datetime(spy_data['Date'])

# Merge the datasets on the date columns
merged_data = pd.merge(news_sentiment, spy_data, left_on='date', right_on='Date')

# Plot the data with secondary y-axis
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot News Sentiment on primary y-axis
ax1.set_xlabel('Date')
ax1.set_ylabel('News Sentiment', color='blue')
ax1.plot(merged_data['date'], merged_data['News Sentiment'], color='blue', label='News Sentiment')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a secondary y-axis for SPY closing prices
ax2 = ax1.twinx()
ax2.set_ylabel('SPY Close', color='red')
ax2.plot(merged_data['date'], merged_data['Close'], color='red', label='SPY Close')
ax2.tick_params(axis='y', labelcolor='red')

# Add title and show plot
plt.title('News Sentiment vs. SPY Closing Prices')
fig.tight_layout()  # To ensure the labels do not overlap
plt.show()
