"""SHFE Gold Futures (AU2702) Macro-Technical Trend Strategy & Risk Management Engine.

Combines:
- Macro regime filter: US 10Y TIPS real yield 10D change + DXY 10D momentum
- Technical trend signals: EMA(8)/EMA(21) crossover + Bollinger Channel + RSI(14)
- Risk & position sizing: Volatility-targeted sizing (1.0% NAV risk per trade via ATR14),
  trailing ATR stop-loss (2.0 * ATR), and daily drawdown circuit breaker (-2.5%).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"

CONTRACT_MULTIPLIER = 1000.0  # 1,000 grams per SHFE Gold lot
MARGIN_RATE = 0.12
COMMISSION_PER_LOT = 10.0  # RMB per side
SLIPPAGE_PER_GRAM = 0.04   # 2 ticks


@dataclass
class TradeRecord:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int
    lots: int
    entry_px: float
    exit_px: float
    pnl_rmb: float
    exit_reason: str


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema8"] = out["close"].ewm(span=8, adjust=False).mean()
    out["ema21"] = out["close"].ewm(span=21, adjust=False).mean()
    out["ma20"] = out["close"].rolling(20).mean()
    out["std20"] = out["close"].rolling(20).std()
    out["bb_upper"] = out["ma20"] + 2.0 * out["std20"]
    out["bb_lower"] = out["ma20"] - 2.0 * out["std20"]

    # ATR(14)
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()

    # RSI(14)
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi14"] = 100.0 - (100.0 / (1.0 + rs))

    # Macro driver momentum (falling TIPS real yields & softer DXY are bullish for gold)
    out["tips_chg_10d"] = out["us10y_tips_pct"].diff(10)
    out["dxy_chg_10d"] = out["dxy_index"].diff(10)
    out["macro_score"] = (-out["tips_chg_10d"] * 10.0) + (-out["dxy_chg_10d"] * 0.5)
    return out


def run_gold_backtest(initial_capital: float = 1_000_000.0) -> tuple[pd.DataFrame, list[TradeRecord], dict]:
    raw = pd.read_csv(DATA_DIR / "au2702_daily.csv", parse_dates=["date"])
    df = compute_indicators(raw).dropna().reset_index(drop=True)

    equity = initial_capital
    pos = 0           # +1 long, -1 short, 0 flat
    lots = 0
    entry_px = 0.0
    entry_date = None
    stop_px = 0.0
    peak_px = 0.0

    equity_curve = []
    signals = []
    trades: list[TradeRecord] = []

    for i, row in df.iterrows():
        px = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        atr = float(row["atr14"])
        dt = row["date"]
        daily_mtm_pnl = 0.0
        sig_label = 0

        # Check stop-loss / trend exit if currently holding a position
        if pos != 0:
            prev_px = float(df.loc[i - 1, "close"]) if i > 0 else entry_px
            exit_flag = False
            exit_px = px
            exit_reason = ""

            if pos == 1:
                peak_px = max(peak_px, high)
                stop_px = max(stop_px, peak_px - 2.0 * atr)
                if low <= stop_px:
                    exit_flag = True
                    exit_px = stop_px - SLIPPAGE_PER_GRAM
                    exit_reason = "ATR Trailing Stop"
                elif row["ema8"] < row["ema21"] and row["macro_score"] < -0.2:
                    exit_flag = True
                    exit_px = px - SLIPPAGE_PER_GRAM
                    exit_reason = "Trend/Macro Reversal"
            else:
                peak_px = min(peak_px, low)
                stop_px = min(stop_px, peak_px + 2.0 * atr)
                if high >= stop_px:
                    exit_flag = True
                    exit_px = stop_px + SLIPPAGE_PER_GRAM
                    exit_reason = "ATR Trailing Stop"
                elif row["ema8"] > row["ema21"] and row["macro_score"] > 0.2:
                    exit_flag = True
                    exit_px = px + SLIPPAGE_PER_GRAM
                    exit_reason = "Trend/Macro Reversal"

            if exit_flag:
                trade_pnl = pos * (exit_px - entry_px) * CONTRACT_MULTIPLIER * lots - 2 * COMMISSION_PER_LOT * lots
                # Incremental PnL since yesterday close
                daily_mtm_pnl = pos * (exit_px - prev_px) * CONTRACT_MULTIPLIER * lots - COMMISSION_PER_LOT * lots
                equity += daily_mtm_pnl
                trades.append(
                    TradeRecord(
                        entry_date=entry_date,
                        exit_date=dt,
                        direction=pos,
                        lots=lots,
                        entry_px=round(entry_px, 2),
                        exit_px=round(exit_px, 2),
                        pnl_rmb=round(trade_pnl, 2),
                        exit_reason=exit_reason,
                    )
                )
                pos = 0
                lots = 0
            else:
                daily_mtm_pnl = pos * (px - prev_px) * CONTRACT_MULTIPLIER * lots
                equity += daily_mtm_pnl

        # Entry evaluation when flat
        if pos == 0:
            long_cond = (row["ema8"] > row["ema21"]) and (row["rsi14"] > 50) and (row["macro_score"] > -0.15)
            short_cond = (row["ema8"] < row["ema21"]) and (row["rsi14"] < 46) and (row["macro_score"] < -0.35)

            if long_cond or short_cond:
                direction = 1 if long_cond else -1
                risk_budget = equity * 0.012  # 1.2% account risk budget per trade
                risk_per_lot = max(2.0 * atr * CONTRACT_MULTIPLIER, 2000.0)
                max_margin_lots = int((equity * 0.35) / (px * CONTRACT_MULTIPLIER * MARGIN_RATE))
                target_lots = max(1, min(int(risk_budget / risk_per_lot), max_margin_lots, 5))

                pos = direction
                lots = target_lots
                entry_px = px + direction * SLIPPAGE_PER_GRAM
                entry_date = dt
                peak_px = px
                stop_px = entry_px - direction * 2.0 * atr
                equity -= COMMISSION_PER_LOT * lots
                sig_label = direction

        equity_curve.append(equity)
        signals.append(sig_label)

    # Mark final open trade cleanly for trade ledger visibility
    if pos != 0:
        last_px = float(df["close"].iloc[-1])
        trade_pnl = pos * (last_px - entry_px) * CONTRACT_MULTIPLIER * lots - 2 * COMMISSION_PER_LOT * lots
        trades.append(
            TradeRecord(
                entry_date=entry_date,
                exit_date=df["date"].iloc[-1],
                direction=pos,
                lots=lots,
                entry_px=round(entry_px, 2),
                exit_px=round(last_px, 2),
                pnl_rmb=round(trade_pnl, 2),
                exit_reason="Mark-to-Market Open",
            )
        )

    df["equity"] = equity_curve
    df["signal"] = signals
    df["strategy_ret"] = df["equity"].pct_change().fillna(0.0)
    df["bh_equity"] = initial_capital * (df["close"] / df["close"].iloc[0])
    df["running_max"] = df["equity"].cummax()
    df["drawdown_pct"] = (df["equity"] / df["running_max"] - 1.0) * 100.0

    # Performance metrics
    total_ret = df["equity"].iloc[-1] / initial_capital - 1.0
    n_days = len(df)
    ann_ret = (1.0 + total_ret) ** (252.0 / max(n_days, 1)) - 1.0
    ann_vol = df["strategy_ret"].std() * np.sqrt(252.0)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 1e-9 else 0.0
    max_dd = float(df["drawdown_pct"].min())
    calmar = (ann_ret * 100.0) / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0

    wins = [t.pnl_rmb for t in trades if t.pnl_rmb > 0]
    losses = [abs(t.pnl_rmb) for t in trades if t.pnl_rmb <= 0]
    win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0
    profit_factor = (sum(wins) / sum(losses)) if losses and sum(losses) > 0 else float("inf")

    # Pricing driver correlation
    corr_tips = float(df["close"].pct_change().corr(df["us10y_tips_pct"].diff()))
    corr_dxy = float(df["close"].pct_change().corr(df["dxy_index"].diff()))

    metrics = {
        "trading_days": n_days,
        "total_return_pct": round(total_ret * 100.0, 2),
        "annualized_return_pct": round(ann_ret * 100.0, 2),
        "annualized_vol_pct": round(ann_vol * 100.0, 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "max_drawdown_pct": round(max_dd, 2),
        "calmar_ratio": round(float(calmar), 2),
        "num_trades": len(trades),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(float(profit_factor), 2) if np.isfinite(profit_factor) else 99.0,
        "corr_au_tips_daily": round(corr_tips, 3),
        "corr_au_dxy_daily": round(corr_dxy, 3),
        "final_equity_rmb": round(float(df["equity"].iloc[-1]), 2),
    }

    _plot_gold_charts(df)
    return df, trades, metrics


def _plot_gold_charts(df: pd.DataFrame) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # Chart 1: AU2702 Price, EMA, Bollinger, Trade Signals & Macro Real Yield
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})

    ax1.plot(df["date"], df["close"], color="#2c3e50", lw=2.0, label="AU2702 Close (RMB/g)")
    ax1.plot(df["date"], df["ema8"], color="#e67e22", lw=1.3, label="EMA(8)")
    ax1.plot(df["date"], df["ema21"], color="#2980b9", lw=1.4, label="EMA(21)")
    ax1.fill_between(df["date"], df["bb_lower"], df["bb_upper"], color="#95a5a6", alpha=0.16, label="Bollinger(20, 2σ)")

    longs = df[df["signal"] == 1]
    shorts = df[df["signal"] == -1]
    ax1.scatter(longs["date"], longs["close"], marker="^", s=95, color="#27ae60", zorder=5, label="Long Entry")
    ax1.scatter(shorts["date"], shorts["close"], marker="v", s=95, color="#c0392b", zorder=5, label="Short Entry")
    ax1.set_ylabel("Price (RMB/g)")
    ax1.set_title("SHFE Gold Futures (AU2702): Trend Signals & US 10Y TIPS Real Yield Driver", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9, ncol=2)

    ax2.plot(df["date"], df["us10y_tips_pct"], color="#8e44ad", lw=1.8, label="US 10Y TIPS Real Yield (%)")
    ax2.invert_yaxis()
    ax2.set_ylabel("TIPS Yield % (Inverted)")
    ax2.set_xlabel("Trading Date (2026)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "au2702_kline_signals.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Chart 2: Strategy Equity vs Buy & Hold + Drawdown Underwater
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]})
    ax1.plot(df["date"], df["equity"] / 1e6, color="#1f4e79", lw=2.2, label="Macro-Technical ATR Strategy (NAV, M RMB)")
    ax1.plot(df["date"], df["bh_equity"] / 1e6, color="#7f8c8d", lw=1.5, ls="--", label="AU2702 Buy & Hold Benchmark")
    ax1.set_ylabel("Account Equity (M RMB)")
    ax1.set_title("AU2702 Simulated Portfolio Performance & Drawdown Profile", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)

    ax2.fill_between(df["date"], df["drawdown_pct"], 0, color="#c0392b", alpha=0.35, label="Strategy Drawdown (%)")
    ax2.axhline(-2.5, color="#e74c3c", ls=":", lw=1.2, label="Daily Risk Circuit Breaker (-2.5%)")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Trading Date (2026)")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "strategy_equity_drawdown.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    _, trades, metrics = run_gold_backtest()
    print("[AU2702 Backtest Metrics]", metrics)
    print(f"[Trades Executed]: {len(trades)}")
