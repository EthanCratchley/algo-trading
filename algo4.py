import backtrader as bt
import numpy as np
import csv
import os

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

class DuelingMomentumStrategy(bt.Strategy):
    params = (
        ('short_period', 10),  # Short-term momentum period
        ('long_period', 50),  # Long-term momentum period
        ('exit_after', 15),  # Exit after x bars
        ('stop_loss', 0.02),  # Stop loss as a percentage
        ('take_profit', 0.05),  # Take profit as a percentage
    )

    def __init__(self):
        self.short_momentum = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.short_period)
        self.long_momentum = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.long_period)
        self.crossover_up = bt.indicators.CrossUp(self.short_momentum, self.long_momentum)
        self.crossover_down = bt.indicators.CrossDown(self.short_momentum, self.long_momentum)
        self.buy_order = None
        self.sell_order = None
        self.bar_executed = 0
        self.total_trades = 0
        self.total_commissions = 0
        self.total_wins = 0
        self.total_losses = 0
        self.total_profit = 0
        self.total_loss = 0
        self.initial_value = 0
        self.final_value = 0
        self.stop_loss_price = None
        self.take_profit_price = None

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
        if not self.position:  # Not in the market
            size = self.broker.get_cash() * 0.1 // self.data.close[0]
            if self.crossover_up:
                self.buy_order = self.buy(size=size)
                self.bar_executed = len(self)
                self.stop_loss_price = self.data.close[0] * (1 - self.params.stop_loss)
                self.take_profit_price = self.data.close[0] * (1 + self.params.take_profit)
            elif self.crossover_down:
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

    def start(self):
        self.initial_value = self.broker.getvalue()

    def stop(self):
        self.final_value = self.broker.getvalue()
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
        annual_return = np.mean(self.returns) * 252  # Assuming 252 trading days in a year
        max_drawdown = max(self.drawdowns)
        calmar_ratio = annual_return / max_drawdown if max_drawdown else 1
        return calmar_ratio

if __name__ == '__main__':
    # List of securities
    securities = ['AAON', 'AEO', 'ASB', 'AVGO', 'CCJ', 'CEG', 'DELL', 'ESTC', 'MU', 'NVT', 'PAAS', 'SNV', 'TDS', 'TSM', 'VLO', 'VST']

    # Create paths for securities and performance folders
    securities_folder = 'securities'
    performance_folder = 'performance4'
    os.makedirs(performance_folder, exist_ok=True)

    for security in securities:
        data_file = os.path.join(securities_folder, f'{security}.csv')
        if not os.path.exists(data_file):
            print(f"Data file for {security} not found. Skipping.")
            continue
        data = bt.feeds.GenericCSVData(
            dataname=os.path.join(securities_folder, f'{security}.csv'),
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
            DuelingMomentumStrategy,
            short_period=range(3, 36, 2),
            long_period=range(37, 101, 3),
            exit_after=range(2, 36, 2),
            stop_loss=[0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09],
            take_profit=[0.03, 0.05, 0.07, 0.09]
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
        csv_file = os.path.join(performance_folder, f'4strategy_performance_{security}.csv')
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Short Period', 'Long Period', 'Exit After', 'Stop Loss', 'Take Profit', 'Starting Value', 'Ending Value', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown', 'Total Return', 'Annualized Return', 'Cumulative Return', 'Total Commission Costs', 'Total Trades', 'Win Rate', 'Average Trade Payoff Ratio'])

        # Run the optimization
        results = cerebro.run(maxcpus=1)  # Use all available CPUs

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
                    strat[0].params.short_period,
                    strat[0].params.long_period,
                    strat[0].params.exit_after,
                    strat[0].params.stop_loss,
                    strat[0].params.take_profit,
                    initial_value,
                    final_value,
                    sharpe_ratio['sharperatio'],
                    sortino_ratio,
                    calmar_ratio,
                    drawdown.max.drawdown,
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
            print(f'Best Strategy Parameters for {security} - Short Period: {best_strat.params.short_period}, Long Period: {best_strat.params.long_period}, Exit After: {best_strat.params.exit_after}, Stop Loss: {best_strat.params.stop_loss}, Take Profit: {best_strat.params.take_profit}')
            print('Starting Portfolio Value: %.2f' % best_strat.initial_value)
            print('Ending Portfolio Value: %.2f' % best_strat.final_value)
            print('Sharpe Ratio:', sharpe_ratio['sharperatio'])
            print('Sortino Ratio:', sortino_ratio)
            print('Calmar Ratio:', calmar_ratio)
            print('Max Drawdown:', drawdown.max.drawdown)
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
                DuelingMomentumStrategy,
                short_period=best_strat.params.short_period,
                long_period=best_strat.params.long_period,
                exit_after=best_strat.params.exit_after,
                stop_loss=best_strat.params.stop_loss,
                take_profit=best_strat.params.take_profit
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
            #cerebro_best.plot(volume=False)
        else:
            print(f"No valid strategy found for {security}.")

# securities = ['AAON', 'AEO', 'ASB', 'AVGO', 'CCJ', 'CEG', 'DELL', 'ESTC', 'MU', 'NVT', 'PAAS', 'SNV', 'TDS', 'TSM', 'VLO', 'VST']
