import backtrader as bt
import numpy as np
import math

class KellyCriterionSizer(bt.Sizer):
    def __init__(self):
        self.win_rate = 0.60  # Initial estimated win rate
        self.payoff_ratio = 1.5  # Initial estimated payoff ratio
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

class PriceSpikeStrategy(bt.Strategy):
    params = (
        ('trail_percent', 0.05),  
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.buy_signal = False
        self.short_signal = False
        self.trade_returns = []
        self.total_trades = 0
        self.total_commissions = 0
        self.total_wins = 0
        self.total_losses = 0
        self.total_profit = 0
        self.total_loss = 0
        self.initial_value = 0
        self.final_value = 0
        self.buy_price = None
        self.sell_price = None

    def next(self):
        if len(self.dataclose) < 2:
            return

        # Entry conditions
        if self.dataclose[0] >= self.dataclose[-1] * 1.05:
            self.buy_signal = True
            self.short_signal = False
            print(f"Buy signal on {self.datas[0].datetime.date(0)} at price {self.dataclose[0]}")
        elif self.dataclose[0] <= self.dataclose[-1] * 0.95:
            self.buy_signal = False
            self.short_signal = True
            print(f"Sell signal on {self.datas[0].datetime.date(0)} at price {self.dataclose[0]}")
        else:
            self.buy_signal = False
            self.short_signal = False

        # Execute orders based on signals
        if not self.position:
            size = self.broker.get_cash() * 0.1 // self.dataclose[0]
            if self.buy_signal:
                self.buy_price = self.dataclose[0]
                self.order = self.buy(size=size)
                print(f"Buy order placed at {self.dataclose[0]} for size {size}")
            elif self.short_signal:
                self.sell_price = self.dataclose[0]
                self.order = self.sell(size=size)
                print(f"Sell order placed at {self.dataclose[0]} for size {size}")
        else:
            # Exit conditions for long position
            if self.position.size > 0:
                if self.dataclose[0] <= self.buy_price * (1 - self.params.trail_percent) or self.dataclose[0] >= self.buy_price * (1 + self.params.trail_percent):
                    self.close()
                    print(f"Closing long position at {self.dataclose[0]}")
            # Exit conditions for short position
            elif self.position.size < 0:
                if self.dataclose[0] >= self.sell_price * (1 + self.params.trail_percent) or self.dataclose[0] <= self.sell_price * (1 - self.params.trail_percent):
                    self.close()
                    print(f"Closing short position at {self.dataclose[0]}")

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
            self.trade_returns.append(pnl / trade.price)

    def start(self):
        self.initial_value = self.broker.getvalue()

    def stop(self):
        self.final_value = self.broker.getvalue()
        if self.trade_returns:
            average_return = sum(self.trade_returns) / len(self.trade_returns)
            downside_returns = [r for r in self.trade_returns if r < 0]
            downside_deviation = math.sqrt(sum(r**2 for r in downside_returns) / len(downside_returns)) if downside_returns else 0
            sortino_ratio = average_return / downside_deviation if downside_deviation != 0 else None
        else:
            sortino_ratio = None
        self.sortino_ratio = sortino_ratio
        self.win_rate = self.total_wins / self.total_trades if self.total_trades > 0 else 0
        self.average_payoff = (self.total_profit / self.total_wins) / (self.total_loss / self.total_losses) if self.total_wins > 0 and self.total_losses > 0 else 0

class SortinoRatio(bt.Analyzer):
    def create_analysis(self):
        self.rets = []

    def notify_trade(self, trade):
        if trade.isclosed:
            self.rets.append(trade.pnlcomm / self.strategy.broker.getvalue())

    def get_analysis(self):
        if not self.rets:
            return float('nan')
        neg_rets = [r for r in self.rets if r < 0]
        sortino_ratio = np.mean(self.rets) / (np.std(neg_rets) if neg_rets else 1)
        return sortino_ratio

class CalmarRatio(bt.Analyzer):
    def __init__(self):
        self.drawdowns = []
        self.returns = []

    def notify_trade(self, trade):
        if trade.isclosed:
            self.returns.append(trade.pnlcomm / self.strategy.broker.getvalue())
            self.drawdowns.append(self.strategy.analyzers.drawdown.get_analysis().max.drawdown)

    def get_analysis(self):
        if not self.returns or not self.drawdowns:
            return float('nan')
        annual_return = np.mean(self.returns) * 252  # Assuming 252 trading days in a year
        max_drawdown = max(self.drawdowns) if self.drawdowns else float('nan')
        calmar_ratio = annual_return / max_drawdown if max_drawdown else float('nan')
        return calmar_ratio

if __name__ == '__main__':
    cerebro = bt.Cerebro()

    data = bt.feeds.GenericCSVData(
        dataname='AMD.csv',
        dtformat=('%Y-%m-%d'),
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1
    )

    cerebro.adddata(data)
    cerebro.addstrategy(PriceSpikeStrategy)
    cerebro.addsizer(KellyCriterionSizer)
    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.001)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(SortinoRatio, _name='sortino')
    cerebro.addanalyzer(CalmarRatio, _name='calmar')

    print('Ticker: TSLA')
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Get analyzers
    strat = results[0]
    sharpe_ratio = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    sortino_ratio = strat.analyzers.sortino.get_analysis()
    calmar_ratio = strat.analyzers.calmar.get_analysis()

    # Print the performance metrics
    print('-----------------------------------------------------------------')
    print('Sharpe Ratio:', sharpe_ratio['sharperatio'])
    print('Sortino Ratio:', sortino_ratio)
    print('Calmar Ratio:', calmar_ratio)
    print('-----------------------------------------------------------------')
    print('Max Drawdown:', drawdown.max.drawdown)
    print('Max Drawdown Duration:', drawdown.max.len)
    print('-----------------------------------------------------------------')
    print('Total Return:', returns['rtot'])
    print('Annualized Return:', returns['rnorm'])
    print('Cumulative Return:', (cerebro.broker.getvalue() - 10000) / 10000)
    print('-----------------------------------------------------------------')
    # Print additional performance metrics
    print('Total Commission Costs:', strat.total_commissions)
    print('Total Trades:', strat.total_trades)
    print('Win Rate:', strat.win_rate)
    print('Average Trade Payoff Ratio:', strat.average_payoff)

    # Plot the result with buy/sell markers
    cerebro.plot(volume=False)
