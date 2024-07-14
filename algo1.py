import backtrader as bt

class KellyCriterionSizer(bt.Sizer):
    def __init__(self):
        self.win_rate = 0.50  # Initial estimated win rate
        self.payoff_ratio = 1.30  # Initial estimated payoff ratio
        self.kelly_fraction = self.calculate_kelly_fraction()

    def calculate_kelly_fraction(self):
        # Kelly formula
        W = self.win_rate
        R = self.payoff_ratio
        kelly_fraction = W - (1 - W) / R
        return max(0, min(kelly_fraction, 1))  # Bound between 0 and 1

    def _getsizing(self, comminfo, cash, data, isbuy):
        # Calculate position size using Kelly Criterion
        size = (cash * self.kelly_fraction) // data.close[0]
        return size

class ImprovedStopBreakoutStrategy(bt.Strategy):
    params = (
        ('lookback', 20),
        ('exit_after', 10),  # Exit after x bars
        ('stop_loss', 0.02),  # Stop loss as a percentage
        ('take_profit', 0.05),  # Take profit as a percentage
    )

    def __init__(self):
        self.highest_high = bt.indicators.Highest(self.data.high(-1), period=self.params.lookback)
        self.lowest_low = bt.indicators.Lowest(self.data.low(-1), period=self.params.lookback)
        self.buy_order = None
        self.sell_order = None
        self.bar_executed = 0  # To track when the order was executed
        self.total_trades = 0
        self.total_commissions = 0
        self.total_wins = 0
        self.total_losses = 0
        self.total_profit = 0
        self.total_loss = 0

    def notify_trade(self, trade):
        if trade.justopened:
            self.total_trades += 1
        if trade.isclosed:
            self.total_commissions += trade.commission
            pnl = trade.pnlcomm
            if pnl > 0:
                self.total_wins += 1
                self.total_profit += pnl
            else:
                self.total_losses += 1
                self.total_loss += abs(pnl)

    def next(self):
        if len(self.data) < self.params.lookback:
            return  # Not enough data for the lookback period

        if not self.position:  # Not in the market
            size = self.broker.get_cash() * 0.1 // self.data.close[0]
            if self.data.close[0] >= self.highest_high[0]:
                self.buy_order = self.buy(size=size)
                self.bar_executed = len(self)
                self.stop_loss_price = self.data.close[0] * (1 - self.params.stop_loss)
                self.take_profit_price = self.data.close[0] * (1 + self.params.take_profit)
            elif self.data.close[0] <= self.lowest_low[0]:
                self.sell_order = self.sell(size=size)
                self.bar_executed = len(self)
                self.stop_loss_price = self.data.close[0] * (1 + self.params.stop_loss)
                self.take_profit_price = self.data.close[0] * (1 - self.params.take_profit)
        else:
            if self.position.size > 0:  # Long position
                if self.data.close[0] <= self.stop_loss_price or self.data.close[0] >= self.take_profit_price:
                    self.close()
                elif len(self) >= self.bar_executed + self.params.exit_after:
                    self.close()
            elif self.position.size < 0:  # Short position
                if self.data.close[0] >= self.stop_loss_price or self.data.close[0] <= self.take_profit_price:
                    self.close()
                elif len(self) >= self.bar_executed + self.params.exit_after:
                    self.close()

    def stop(self):
        self.win_rate = self.total_wins / self.total_trades if self.total_trades > 0 else 0
        self.average_payoff = (self.total_profit / self.total_wins) / (self.total_loss / self.total_losses) if self.total_wins > 0 and self.total_losses > 0 else 0

if __name__ == '__main__':
    # Load data
    data = bt.feeds.GenericCSVData(
        dataname='fit5.csv',
        dtformat=('%Y-%m-%d'),
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=6,
        openinterest=-1  # -1 indicates no open interest column in CSV
    )

    # Create Cerebro engine
    cerebro = bt.Cerebro()

    # Add data to Cerebro
    cerebro.adddata(data)

    # Add strategy to Cerebro
    cerebro.addstrategy(ImprovedStopBreakoutStrategy)

    # Add Kelly Criterion Sizer to Cerebro
    cerebro.addsizer(KellyCriterionSizer)

    # Set our desired cash start
    cerebro.broker.set_cash(10000.0)

    # Set the commission
    cerebro.broker.setcommission(commission=0.002)

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

    # Print additional performance metrics
    print('Total Trades:', strat.total_trades)
    print('Total Commission Costs:', strat.total_commissions)
    print('Win Rate:', strat.win_rate)
    print('Average Trade Payoff Ratio:', strat.average_payoff)

    # Plot the result, skip volume plotting if not available
    cerebro.plot(volume=False)
