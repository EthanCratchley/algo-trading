import pandas as pd

# Load the news sentiment data
news_sentiment = pd.read_csv('news_sentiment_updated.csv')

# Load the SPY data
spy_data = pd.read_csv('SPY (1).csv')

# Convert the date columns to datetime format
news_sentiment['date'] = pd.to_datetime(news_sentiment['date'])
spy_data['Date'] = pd.to_datetime(spy_data['Date'])

# Check for missing dates in News Sentiment data that are present in SPY data
missing_dates = spy_data[~spy_data['Date'].isin(news_sentiment['date'])]['Date']

# Create a DataFrame for the missing dates with a placeholder sentiment value
missing_data = pd.DataFrame({'date': missing_dates, 'News Sentiment': 0.0})

# Append the missing data to the news sentiment data
news_sentiment = pd.concat([news_sentiment, missing_data])

# Remove any duplicates that may have been introduced
news_sentiment = news_sentiment.drop_duplicates(subset=['date'])

# Filter the news sentiment data to only include dates that are present in the SPY data
news_sentiment = news_sentiment[news_sentiment['date'].isin(spy_data['Date'])]

# Sort the DataFrame by date
news_sentiment = news_sentiment.sort_values(by='date').reset_index(drop=True)

# Save the updated DataFrame back to a CSV file
news_sentiment.to_csv('news_sentiment_final.csv', index=False)

print(f"Number of rows in updated News Sentiment data: {len(news_sentiment)}")
print(f"Number of rows in SPY data: {len(spy_data)}")
