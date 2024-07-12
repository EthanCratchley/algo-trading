import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt

# Define the trading strategy
class VXXStrategy(bt.Strategy):
    params = (
        ('short_percentage', 0.05),
        ('buyback_percentage', 0.3333),
        ('interval_days', 15),
        ('threshold_gain', 0.20),
        ('lookback_days', 25),  # Added parameter for lookback period
    )
    
    def __init__(self):
        self.short_positions = []
        self.cash_start = self.broker.get_cash()
        self.val_start = self.broker.get_value()
        
    def next(self):
        if len(self.data) <= self.params.lookback_days:
            return

        current_price = self.data.close[0]
        lookback_price = self.data.close[-self.params.lookback_days]
        gain = (current_price - lookback_price) / lookback_price

        # Shorting condition
        if gain >= self.params.threshold_gain:
            short_amount = self.broker.get_cash() * self.params.short_percentage
            size = short_amount // current_price
            if size > 0:
                self.sell(size=size)
                self.short_positions.append({'date': self.data.datetime.date(0), 'price': current_price, 'size': size})
                print(f"Shorting on {self.data.datetime.date(0)}: {size} shares at ${current_price:.2f}")
                
                # Schedule buyback
                buyback_dates = [self.data.datetime.date(0) + pd.Timedelta(days=self.params.interval_days * j) for j in range(1, 4)]
                for bd in buyback_dates:
                    self.short_positions.append({'date': bd, 'size': size * self.params.buyback_percentage, 'action': 'buy'})

        # Buyback condition
        for pos in self.short_positions:
            if pos['date'] == self.data.datetime.date(0) and pos.get('action') == 'buy':
                size = pos['size']
                if size > 0:
                    self.buy(size=size)
                    print(f"Buyback on {self.data.datetime.date(0)}: {size} shares at ${current_price:.2f}")

# Load VXX historical data
vxx_data = pd.read_csv('vxx_data.csv', parse_dates=True, index_col='Date')

# Set up the backtesting environment
cerebro = bt.Cerebro()
cerebro.addstrategy(VXXStrategy)

# Convert the DataFrame to a DataFeed
data = bt.feeds.PandasData(dataname=vxx_data)

cerebro.adddata(data)

# Set initial cash
cerebro.broker.set_cash(10000)

# Set commission (optional)
cerebro.broker.setcommission(commission=0.001)

# Add analyzers
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

# Run the backtest
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
results = cerebro.run()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Get analyzers
sharpe_ratio = results[0].analyzers.sharpe.get_analysis()
drawdown = results[0].analyzers.drawdown.get_analysis()
returns = results[0].analyzers.returns.get_analysis()

# Print performance metrics
print(f"Sharpe Ratio: {sharpe_ratio['sharperatio']:.2f}")
print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")
print(f"Annual Return: {returns['rnorm']:.2f}%")
print(f"Cumulative Return: {returns['rtot']:.2f}%")

# Plot the results
cerebro.plot(style='candlestick')
