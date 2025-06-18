import asyncio
import ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import logging
import json
import sys
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.dates as mdates

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if 'debug' in sys.argv else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hybrid_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HybridEngine")

class HybridTradingEngine:
    def __init__(self, config, backtest=False, debug=False):
        self.config = config
        self.is_backtest = backtest
        self.debug = debug

        if self.is_backtest:
            self.exchange = ccxt.kucoin()
        else:
            self.exchange = self._initialize_exchange()

        self.capital = config['initial_capital']
        self.active_positions = {}
        self.strategy_state = {
            'last_trade_time': {},
            'session': 'asian',
            'regime': 'neutral',
            'symbol_priority': {} if self.is_backtest else self._calculate_symbol_priority()
        }
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0
        }
        self.trade_log = []

    def _initialize_exchange(self):
        exchange = ccxt.binance({
            'apiKey': self.config['api_key'],
            'secret': self.config['api_secret'],
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
        })
        exchange.load_markets()
        return exchange


    def _calculate_symbol_priority(self):
        symbols = self.config['symbols']
        priorities = {}
        for symbol in symbols:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, '5m', limit=24)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                volatility = (df['high'].max() - df['low'].min()) / df['close'].mean()
                volume_ratio = df['volume'].iloc[-1] / df['volume'].mean()
                priorities[symbol] = volatility * volume_ratio
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                priorities[symbol] = 0
        max_priority = max(priorities.values()) or 1
        return {k: v / max_priority for k, v in sorted(priorities.items(), key=lambda item: item[1], reverse=True)}


    def visualize_trades(self, df):
            try:
                trades = pd.DataFrame(self.trade_log)

                if trades.empty:
                    logger.warning("No trades to visualize.")
                    return

                # Ensure timestamps match
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                trades['timestamp'] = pd.to_datetime(trades['timestamp'])

                mpf_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
                trade_markers = []

                for _, trade in trades.iterrows():
                    ts = trade['timestamp']
                    price = float(trade['entry_price'])
                    action = trade['action']
                    color = 'green' if trade['profit'] > 0 else 'red'
                    marker = '^' if action == 'buy' else 'v'

                    if ts not in mpf_df.index:
                        logger.warning(f"Trade timestamp {ts} not in OHLCV index. Skipping.")
                        continue

                    # Build array with float and None only
                    marker_data = np.full(len(mpf_df), np.nan)
                    idx = mpf_df.index.get_loc(ts)
                    marker_data[idx] = price

                    trade_markers.append(
                        mpf.make_addplot(
                            marker_data,
                            type='scatter',
                            marker=marker,
                            markersize=100,
                            color=color
                        )
                    )

                # Plot it
                mpf.plot(
                    mpf_df,
                    type='candle',
                    style='yahoo',
                    title='Trade Chart',
                    volume=True,
                    addplot=trade_markers,
                    figratio=(16, 8),
                    figscale=1.2
                )

            except Exception as e:
                logger.error(f"Failed to visualize trades: {str(e)}")

    def backtest(self, symbol, timeframe='5m', lookback_days=15):
        try:
            since = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000)
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, params={})
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('US/Pacific')

            trades = 0
            wins = 0
            pnl = 0

            for i in range(50, len(df) - 6):
                subset = df.iloc[i-50:i].copy()
                ticker_mock = {'quoteVolume': subset['volume'].iloc[-1]}
                signal = self._core_strategy_mock(subset, ticker_mock)
                if signal:
                    entry_time = df['timestamp'].iloc[i]
                    entry_price = subset['close'].iloc[-1]
                    future_window = df.iloc[i+1:i+6]
                    hit_target = False
                    profit = 0
                    exit_time = df['timestamp'].iloc[i+5]  # default to last candle
                    exit_reason = 'timeout'

                    atr = subset['atr'].iloc[-1]
                    risk_multiple = 1.5  # Target is 1.5x ATR; Stop is 1.0x ATR

                    if signal['action'] == 'buy':
                        target_price = entry_price + atr * risk_multiple
                        stop_price = entry_price - atr
                        for _, row in future_window.iterrows():
                            if row['high'] >= target_price:
                                profit = target_price - entry_price
                                exit_time = row['timestamp']
                                exit_reason = 'target'
                                hit_target = True
                                break
                            elif row['low'] <= stop_price:
                                profit = stop_price - entry_price
                                exit_time = row['timestamp']
                                exit_reason = 'stop'
                                break
                    else:
                        target_price = entry_price - atr * risk_multiple
                        stop_price = entry_price + atr
                        for _, row in future_window.iterrows():
                            if row['low'] <= target_price:
                                profit = entry_price - target_price
                                exit_time = row['timestamp']
                                exit_reason = 'target'
                                hit_target = True
                                break
                            elif row['high'] >= stop_price:
                                profit = entry_price - stop_price
                                exit_time = row['timestamp']
                                exit_reason = 'stop'
                                break

                    if self.debug:
                        logger.debug(
                            f"Simulated {signal['action']} at {entry_price:.2f}, "
                            f"target {target_price:.2f}, stop {stop_price:.2f}, exit: {exit_reason}"
                        )

                    pnl += profit
                    trades += 1
                    if hit_target:
                        wins += 1

                    self.trade_log.append({
                        'timestamp': entry_time,
                        'exit_time': exit_time,
                        'symbol': symbol,
                        'action': signal['action'],
                        'entry_price': entry_price,
                        'target_price': target_price,
                        'stop_price': stop_price,
                        'exit_reason': exit_reason,
                        'hit_target': hit_target,
                        'profit': profit
                    })

            win_rate = wins / trades if trades > 0 else 0
            avg_pnl = pnl / trades if trades > 0 else 0

            logger.info(f"Backtest completed | Trades: {trades} | Win Rate: {win_rate:.2%} | Avg PnL: {avg_pnl:.4f}")

            # Export trade log to CSV
            pd.DataFrame(self.trade_log).to_csv("backtest_trades.csv", index=False)
            #self.visualize_trades(df)

        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")

    def _core_strategy_mock(self, df, ticker):
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['atr'] = self._calculate_atr(df, 14)
        current = df.iloc[-1]

        score_buy = 0
        score_sell = 0

        # Trend Filter
        price_range = df['high'].max() - df['low'].min()
        if price_range / df['close'].mean() < 0.01:
            return None  # Sideways market, skip

        # Buy Conditions
        if current['close'] > current['ma20']:
            score_buy += 1
        if current['ma20'] > current['ma50']:
            score_buy += 1
        if current['rsi'] < 30:
            score_buy += 1
        if ticker['quoteVolume'] > df['volume'].mean() * 1.5 and current['close'] > current['ma20']:
            score_buy += 1

        # Sell Conditions
        if current['close'] < current['ma20']:
            score_sell += 1
        if current['ma20'] < current['ma50']:
            score_sell += 1
        if current['rsi'] > 70:
            score_sell += 1
        if ticker['quoteVolume'] > df['volume'].mean() * 1.5 and current['close'] < current['ma20']:
            score_sell += 1

        if self.debug:
            timestamp = df['timestamp'].iloc[-1]
            logger.debug(f"[{timestamp}] Buy Score: {score_buy}/4 | Sell Score: {score_sell}/4")

        if self.is_backtest:
            if score_buy >= 3 and score_buy > score_sell:
                return {'symbol': 'backtest_symbol', 'action': 'buy', 'confidence': score_buy / 4.0}
            elif score_sell >= 3 and score_sell > score_buy:
                return {'symbol': 'backtest_symbol', 'action': 'sell', 'confidence': score_sell / 4.0}

        return None

    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, df, period=14):
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()

if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)

    backtest_mode = len(sys.argv) > 1 and sys.argv[1] == "backtest"
    debug_mode = len(sys.argv) > 2 and sys.argv[2] == "debug"

    engine = HybridTradingEngine(config, backtest=backtest_mode, debug=debug_mode)

    if backtest_mode:
        engine.backtest(symbol="SOL/USDT")
    else:
        asyncio.run(engine.run())
