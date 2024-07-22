import backtrader as bt
import pandas as pd
from datetime import datetime
import numpy as np

# Custom PandasData feed to include news_sentiment
class CustomPandasData(bt.feeds.PandasData):
    lines = ('news_sentiment',)
    params = (('news_sentiment', 8),)  # assuming 'News Sentiment' is the 9th column in the dataframe (0-indexed)
    
    # Plot configuration
    plotinfo = dict(subplot=True)  # This will plot the news_sentiment on a separate subplot
    plotlines = dict(news_sentiment=dict(_name='News Sentiment', color='blue', linewidth=1.5))

# Create a subclass of Strategy to define the strategy
class NewsSentimentStrategy(bt.Strategy):
    def __init__(self):
        self.news_sentiment = self.datas[0].news_sentiment

    def next(self):
        # Debug: Print current news sentiment value
        print(f'Date: {self.data.datetime.date(0)}, News Sentiment: {self.news_sentiment[0]}')

        if not self.position:  # Not in the market
            if self.news_sentiment[0] <= -0.175:  # Threshold for buying
                self.buy()
                print(f'Buy order created: Date: {self.data.datetime.date(0)}, News Sentiment: {self.news_sentiment[0]}, Price: {self.data.close[0]}')
            elif self.news_sentiment[0] >= 0.175:  # Threshold for selling
                self.sell()
                print(f'Sell order created: Date: {self.data.datetime.date(0)}, News Sentiment: {self.news_sentiment[0]}, Price: {self.data.close[0]}')
        else:  # In the market
            if self.position.size > 0 and self.news_sentiment[0] >= 0.075:  # Long position and sentiment is non-negative
                self.close()
                print(f'Close long position: Date: {self.data.datetime.date(0)}, News Sentiment: {self.news_sentiment[0]}, Price: {self.data.close[0]}')
            elif self.position.size < 0 and self.news_sentiment[0] <= 0.075:  # Short position and sentiment is non-positive
                self.close()
                print(f'Close short position: Date: {self.data.datetime.date(0)}, News Sentiment: {self.news_sentiment[0]}, Price: {self.data.close[0]}')

# Load the data
news_sentiment_data = pd.read_csv('news_sentiment_updated.csv')
spy_data = pd.read_csv('SPY (1).csv')

# Convert the date columns to datetime format
news_sentiment_data['date'] = pd.to_datetime(news_sentiment_data['date'])
spy_data['Date'] = pd.to_datetime(spy_data['Date'])

# Merge the datasets on the date columns
merged_data = pd.merge(news_sentiment_data, spy_data, left_on='date', right_on='Date')

# Prepare the data for Backtrader
data = CustomPandasData(
    dataname=merged_data,
    datetime='date',
    open='Open',
    high='High',
    low='Low',
    close='Close',
    volume='Volume',
    openinterest=-1,
    news_sentiment='News Sentiment'
)

# Initialize the cerebro engine
cerebro = bt.Cerebro()

# Add the data to the engine
cerebro.adddata(data)

# Add the strategy to the engine
cerebro.addstrategy(NewsSentimentStrategy)

# Set our desired cash start
cerebro.broker.setcash(100000.0)

# Set the commission
cerebro.broker.setcommission(commission=0.001)

# Add analyzers for additional statistics
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

# Run over everything
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
results = cerebro.run()
print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Get the analyzers
sharpe = results[0].analyzers.sharpe.get_analysis()
drawdown = results[0].analyzers.drawdown.get_analysis()
trades = results[0].analyzers.trades.get_analysis()

# Print Sharpe ratio
print(f"Sharpe Ratio: {sharpe['sharperatio']}")

# Print drawdown
print("Drawdown:")
print(f"  Max Drawdown: {drawdown['max']['drawdown']:.2f}%")
print(f"  Money Lost: ${drawdown['max']['moneydown']:.2f}")

# Print trade analysis
print("Trade Analysis:")
print(f"  Total Trades: {trades.total.total}")
print(f"  Total Won: {trades.won.total}")
print(f"  Total Lost: {trades.lost.total}")
print(f"  Win Ratio: {trades.won.total / trades.total.total:.2f}")
print(f"  Payoff Ratio: {trades.won.pnl.total / abs(trades.lost.pnl.total):.2f}")
print(f"  Average Win: {trades.won.pnl.average:.2f}")
print(f"  Average Loss: {trades.lost.pnl.average:.2f}")

# Plot the result
cerebro.plot()


