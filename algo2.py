import backtrader as bt
import numpy as np
import math
import os
import csv

class KellyCriterionSizer(bt.Sizer):
    def __init__(self):
        self.win_rate = 0.55  # Initial estimated win rate
        self.payoff_ratio = 1.55  # Initial estimated payoff ratio
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

class SimplePricePattern(bt.Strategy):
    params = (
        ('up_days', 3),
        ('down_days', 3),
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
        if len(self.dataclose) < max(self.params.up_days, self.params.down_days):
            return

        if all(self.dataclose[-i] > self.dataclose[-i-1] for i in range(1, self.params.up_days + 1)):
            self.buy_signal = True
            self.short_signal = False
        elif all(self.dataclose[-i] < self.dataclose[-i-1] for i in range(1, self.params.down_days + 1)):
            self.buy_signal = False
            self.short_signal = True
        else:
            self.buy_signal = False
            self.short_signal = False

        if not self.position:
            size = self.broker.get_cash() * 0.1 // self.dataclose[0]
            if self.buy_signal:
                self.buy_price = self.dataclose[0]
                self.order = self.buy(size=size)
            elif self.short_signal:
                self.sell_price = self.dataclose[0]
                self.order = self.sell(size=size)
        else:
            if self.position.size > 0:  # Long position
                if self.dataclose[0] <= self.buy_price * (1 - self.params.trail_percent) or self.dataclose[0] >= self.buy_price * (1 + self.params.trail_percent):
                    self.close()
            elif self.position.size < 0:  # Short position
                if self.dataclose[0] >= self.sell_price * (1 + self.params.trail_percent) or self.dataclose[0] <= self.sell_price * (1 - self.params.trail_percent):
                    self.close()

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
        neg_rets = [r for r in self.rets if r < 0]
        mean_rets = np.mean(self.rets) if self.rets else 0
        downside_deviation = np.std(neg_rets) if neg_rets else 1
        sortino_ratio = mean_rets / downside_deviation if downside_deviation != 0 else None
        return sortino_ratio

class CalmarRatio(bt.Analyzer):
    def __init__(self):
        self.drawdowns = []
        self.returns = []

    def notify_trade(self, trade):
        if trade.isclosed:
            self.returns.append(trade.pnlcomm / self.strategy.broker.getvalue())
            drawdown = self.strategy.analyzers.drawdown.get_analysis().max.drawdown
            self.drawdowns.append(drawdown)

    def get_analysis(self):
        annual_return = np.mean(self.returns) * 252 if self.returns else 0  # Assuming 252 trading days in a year
        max_drawdown = max(self.drawdowns) if self.drawdowns else 1
        calmar_ratio = annual_return / max_drawdown
        return calmar_ratio

if __name__ == '__main__':
    # Create paths for securities and performance folders
    securities_folder = 'securities'
    performance_folder = 'performance2'
    os.makedirs(performance_folder, exist_ok=True)

    # Load data
    data = bt.feeds.GenericCSVData(
        dataname=os.path.join(securities_folder, 'AAPL.csv'),
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
    cerebro = bt.Cerebro(optreturn=False)

    # Add data to Cerebro
    cerebro.adddata(data)

    # Add strategy to Cerebro with optimization parameters
    cerebro.optstrategy(
        SimplePricePattern,
        up_days=range(3, 22, 1),
        down_days=range(3, 22, 1),
        trail_percent=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    )

    # Add Kelly Criterion Sizer to Cerebro
    cerebro.addsizer(KellyCriterionSizer)

    # Set our desired cash start
    cerebro.broker.set_cash(10000)

    # Set the commission
    cerebro.broker.setcommission(commission=0.002)

    # Add analyzers for performance metrics
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(SortinoRatio, _name='sortino')
    cerebro.addanalyzer(CalmarRatio, _name='calmar')

    # CSV setup
    csv_file = os.path.join(performance_folder, '2strategy_performance_AAPL.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['up_days', 'down_days', 'trail_percent', 'Starting Value', 'Ending Value', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown', 'Total Return', 'Annualized Return', 'Cumulative Return', 'Total Commission Costs', 'Total Trades', 'Win Rate', 'Average Trade Payoff Ratio'])

    # Run the optimization
    results = cerebro.run()  # Adjust maxcpus as needed

    # Track the best strategy
    best_strat = None
    best_sharpe = -np.inf  # Use negative infinity to ensure any valid Sharpe ratio will be better

    for strat in results:
        sharpe_ratio = strat[0].analyzers.sharpe.get_analysis()
        if sharpe_ratio['sharperatio'] is not None and sharpe_ratio['sharperatio'] > best_sharpe:
            best_sharpe = sharpe_ratio['sharperatio']
            best_strat = strat[0]

        # Write each strategy's performance to CSV
        drawdown = strat[0].analyzers.drawdown.get_analysis()
        returns = strat[0].analyzers.returns.get_analysis()
        sortino_ratio = strat[0].analyzers.sortino.get_analysis()
        calmar_ratio = strat[0].analyzers.calmar.get_analysis()
        initial_value = strat[0].initial_value
        final_value = strat[0].final_value

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                strat[0].params.up_days,
                strat[0].params.down_days,
                strat[0].params.trail_percent,
                initial_value,
                final_value,
                sharpe_ratio['sharperatio'],
                sortino_ratio,
                calmar_ratio,
                drawdown.max.drawdown if 'max' in drawdown else None,
                returns['rtot'],
                returns['rnorm'],
                (final_value - initial_value) / initial_value,
                strat[0].total_commissions,
                strat[0].total_trades,
                strat[0].win_rate,
                strat[0].average_payoff
            ])

    # Print out the best strategy's result
    if best_strat is not None:
        sharpe_ratio = best_strat.analyzers.sharpe.get_analysis()
        drawdown = best_strat.analyzers.drawdown.get_analysis()
        returns = best_strat.analyzers.returns.get_analysis()
        sortino_ratio = best_strat.analyzers.sortino.get_analysis()
        calmar_ratio = best_strat.analyzers.calmar.get_analysis()
        
        print('---------------------------------------------------------------')
        print(f'Best Strategy Parameters - up_days: {best_strat.params.up_days}, down_days: {best_strat.params.down_days}, trail_percent: {best_strat.params.trail_percent}')
        print('Starting Portfolio Value: %.2f' % best_strat.initial_value)
        print('Ending Portfolio Value: %.2f' % best_strat.final_value)
        print('Sharpe Ratio:', sharpe_ratio['sharperatio'])
        print('Sortino Ratio:', sortino_ratio)
        print('Calmar Ratio:', calmar_ratio)
        print('Max Drawdown:', drawdown.max.drawdown if 'max' in drawdown else None)
        print('Total Return:', returns['rtot'])
        print('Annualized Return:', returns['rnorm'])
        print('Cumulative Return:', (best_strat.final_value - best_strat.initial_value) / best_strat.initial_value)
        print('Total Commission Costs:', best_strat.total_commissions)
        print('Total Trades:', best_strat.total_trades)
        print('Win Rate:', best_strat.win_rate)
        print('Average Trade Payoff Ratio:', best_strat.average_payoff)
        print('---------------------------------------------------------------')

        # Create a new Cerebro instance for plotting the best strategy
        cerebro_best = bt.Cerebro()

        # Add data to Cerebro
        cerebro_best.adddata(data)

        # Add the best strategy with the best parameters to Cerebro
        cerebro_best.addstrategy(
            SimplePricePattern,
            up_days=best_strat.params.up_days,
            down_days=best_strat.params.down_days,
            trail_percent=best_strat.params.trail_percent
        )

        # Add Kelly Criterion Sizer to Cerebro
        cerebro_best.addsizer(KellyCriterionSizer)

        # Set our desired cash start
        cerebro_best.broker.set_cash(10000)

        # Set the commission
        cerebro_best.broker.setcommission(commission=0.002)

        # Add analyzers for performance metrics
        cerebro_best.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro_best.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro_best.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro_best.addanalyzer(SortinoRatio, _name='sortino')
        cerebro_best.addanalyzer(CalmarRatio, _name='calmar')

        # Run the best strategy
        cerebro_best.run()

        # Plot the result for the best strategy
        cerebro_best.plot(volume=False)
    else:
        print("No valid strategy found.")
