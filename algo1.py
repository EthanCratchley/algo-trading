import backtrader as bt
import pandas as pd

class StopBreakoutStrategy(bt.Strategy):
    params = (
        ('lookback', 20),
        ('exit_after', 10),  # Exit after x bars
    )

    def __init__(self):
        self.highest_high = bt.indicators.Highest(self.data.high(-1), period=self.params.lookback)
        self.lowest_low = bt.indicators.Lowest(self.data.low(-1), period=self.params.lookback)
        self.buy_order = None
        self.sell_order = None
        self.bar_executed = 0  # To track when the order was executed

    def next(self):
        if len(self.data) < self.params.lookback:
            return  # Not enough data for the lookback period

        if not self.position:  # Not in the market
            if self.data.close[0] >= self.highest_high[0]:
                self.buy_order = self.buy()
                self.bar_executed = len(self)
            elif self.data.close[0] <= self.lowest_low[0]:
                self.sell_order = self.sell()
                self.bar_executed = len(self)
        else:
            if self.buy_order and self.data.close[0] <= self.lowest_low[0]:
                self.close()  # Close the long position
                self.sell_order = self.sell()
                self.buy_order = None
                self.bar_executed = len(self)
            elif self.sell_order and self.data.close[0] >= self.highest_high[0]:
                self.close()  # Close the short position
                self.buy_order = self.buy()
                self.sell_order = None
                self.bar_executed = len(self)

            # Exit after specified number of bars
            if len(self) >= self.bar_executed + self.params.exit_after:
                self.close()

if __name__ == '__main__':
    # Load data
    data = bt.feeds.GenericCSVData(
        dataname='fit1.csv',
        dtformat=('%Y-%m-%d'),
        datetime=0,
        high=2,
        low=3,
        open=1,
        close=4,
        volume=6,
        openinterest=-1  # -1 indicates no open interest column in CSV
    )

    # Create Cerebro engine
    cerebro = bt.Cerebro()

    # Add data to Cerebro
    cerebro.adddata(data)

    # Add strategy to Cerebro
    cerebro.addstrategy(StopBreakoutStrategy)

    # Set our desired cash start
    cerebro.broker.set_cash(10000.0)

    # Set the commission
    cerebro.broker.setcommission(commission=0.001)

    # Add analyzers for performance metrics
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    # Run the strategy
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Get analyzers
    strat = results[0]
    sharpe_ratio = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    returns = strat.analyzers.returns.get_analysis()

    # Print the performance metrics
    print('Sharpe Ratio:', sharpe_ratio['sharperatio'])
    print('Max Drawdown:', drawdown.max.drawdown)
    print('Max Drawdown Duration:', drawdown.max.len)
    print('Total Return:', returns['rtot'])
    print('Annualized Return:', returns['rnorm'])
    print('Cumulative Return:', returns['rnorm100'])

    # Plot the result, skip volume plotting if not available
    cerebro.plot(volume=False)
