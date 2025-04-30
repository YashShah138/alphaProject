# ZLMA-Based Bollinger Band Strategy with Kalman Price + Min-Diff Filter (AMD & XLE, 2020-2024)
# --------------------------------------------------------------------------------
# • Bands: ZLMA ± 1.8 σ of Kalman price (window 25).
# • **Buy**  when Kalman crosses up through the LOWER band **and** the gap is ≥ MIN_DIFF_PCT.
# • **Sell** when Kalman crosses down through the UPPER band **and** the gap is ≥ MIN_DIFF_PCT.
# • Gap is measured as |Kal-Band| / Band × 100.
# • Position size 30 %, commission 0.5 %.
# --------------------------------------------------------------------------------

import pandas as pd, numpy as np, yfinance as yf

START_DATE, END_DATE = "2020-01-01", "2024-12-31"
TICKERS = ["AMD", "XLE"]
INITIAL_CAPITAL = 100_000
TRADE_PCT = 0.40
FEE_PCT   = 0.005
MIN_DIFF_PCT = 0.8    # minimum % gap Kalman must clear beyond band
RISK_FREE_RATE, TRADING_DAYS = 0.0, 252

LEN_ZLMA = 25
BB_MULT  = 0.01
KF_Q, KF_R = 0.02, 0.08

# ----------------------------------------------------------------------------
# Kalman filter
# ----------------------------------------------------------------------------

def kalman_filter(series, q=KF_Q, r=KF_R):
    x = np.empty_like(series, dtype="float64"); p=1.0; x_prev=series.iloc[0]; x[0]=x_prev
    for i in range(1, len(series)):
        p += q; k = p / (p + r)
        x_cur = x_prev + k * (series.iloc[i] - x_prev)
        p = (1 - k) * p; x[i] = x_cur; x_prev = x_cur
    return pd.Series(x, index=series.index)

# ----------------------------------------------------------------------------
# Zero-Lag MA
# ----------------------------------------------------------------------------

def zlma(series, length=LEN_ZLMA):
    half = int(round(length / 2))
    ema1 = series.ewm(span=half, adjust=False).mean()
    ema2 = ema1.ewm(span=half, adjust=False).mean()
    return ema1 + (ema1 - ema2)

# ----------------------------------------------------------------------------
# Data & indicators
# ----------------------------------------------------------------------------

def get_prices():
    raw = yf.download(TICKERS, START_DATE, END_DATE, interval="1d", auto_adjust=False, progress=False)
    closes = raw["Adj Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Adj Close"]]
    if not isinstance(closes.columns, pd.MultiIndex): closes.columns = TICKERS
    return closes.dropna(how="all")


def add_indicators(closes):
    df = closes.copy()
    for t in closes.columns:
        kal = kalman_filter(closes[t])
        z   = zlma(kal)
        std = kal.rolling(LEN_ZLMA).std()
        upper = z + BB_MULT * std
        lower = z - BB_MULT * std
        df[f"{t}_kal"], df[f"{t}_z"], df[f"{t}_up"], df[f"{t}_lo"] = kal, z, upper, lower
    return df.dropna()

# ----------------------------------------------------------------------------
# Backtester
# ----------------------------------------------------------------------------

class Backtester:
    def __init__(self, data, cash):
        self.d, self.cash = data, cash
        self.hold = {t: 0 for t in TICKERS}
        self.trades, self.track = [], []

    def _nav(self, d):
        px = self.d.loc[d, TICKERS]
        return self.cash + sum(self.hold[t] * px[t] for t in TICKERS)

    def run(self):
        prev = self.d.shift(1); dates = self.d.index
        self.track.append((dates[0], self._nav(dates[0])))
        for dt in dates[1:]:
            for t in TICKERS:
                kal, kal_y = self.d.at[dt, f"{t}_kal"], prev.at[dt, f"{t}_kal"]
                lo,  lo_y  = self.d.at[dt, f"{t}_lo" ], prev.at[dt, f"{t}_lo" ]
                up,  up_y  = self.d.at[dt, f"{t}_up" ], prev.at[dt, f"{t}_up" ]
                price = self.d.at[dt, t]

                # % gaps
                gap_lo = (kal - lo) / lo * 100 if lo else 0
                gap_up = (up - kal) / up * 100 if up else 0

                # BUY: cross up through lower band with sufficient gap
                buy_cross = (kal_y <= lo_y) and (kal > lo) and (gap_lo >= MIN_DIFF_PCT)
                if self.hold[t] == 0 and buy_cross:
                    spend = self.cash * TRADE_PCT
                    sh = int(spend // (price * (1 + FEE_PCT)))
                    if sh:
                        cost = sh * price; fee = cost * FEE_PCT
                        self.cash -= cost + fee; self.hold[t] = sh
                        self.trades.append(dict(date=dt, ticker=t, act="BUY", sh=sh, px=price, fee=fee))
                    continue  # skip sell check same bar

                # SELL: cross down through upper band with sufficient gap
                sell_cross = (kal_y >= up_y) and (kal < up) and (gap_up >= MIN_DIFF_PCT)
                if self.hold[t] > 0 and sell_cross:
                    sh = self.hold[t]
                    proceeds = sh * price; fee = proceeds * FEE_PCT
                    self.cash += proceeds - fee; self.hold[t] = 0
                    self.trades.append(dict(date=dt, ticker=t, act="SELL", sh=sh, px=price, fee=fee))
            self.track.append((dt, self._nav(dt)))
        return self.track[-1][1]

    def nav_series(self):
        d, v = zip(*self.track); return pd.Series(v, index=d)

# ----------------------------------------------------------------------------
# Sharpe
# ----------------------------------------------------------------------------

def sharpe(nav):
    r = nav.pct_change().dropna(); ex = r - RISK_FREE_RATE / TRADING_DAYS
    return np.nan if ex.std() == 0 else (ex.mean() / ex.std()) * np.sqrt(TRADING_DAYS)

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    prices = get_prices(); data = add_indicators(prices)
    bt = Backtester(data, INITIAL_CAPITAL)
    final = bt.run(); nav = bt.nav_series(); sr = sharpe(nav)

    print("\n====== ZLMA-BB Kalman Strategy (Min-Diff) Backtest ======")
    print(f"Final value: ${final:,.2f}  |  Return: {(final/INITIAL_CAPITAL-1):.2%}  |  Sharpe: {sr:.3f}\n")
    log = pd.DataFrame(bt.trades); print("Trades:" if not log.empty else "No trades.")
    if not log.empty:
        print(log.to_string(index=False))
