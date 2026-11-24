"""Data pipeline for SHFE Cast Aluminum Alloy (AD2612/AD2701) & Gold (AU2702) futures.

Supports two modes:
1. Reproducible offline market dataset (default) calibrated to SHFE contract specs
   for Oct–Nov 2026 training & strategy backtesting.
2. Optional live SHFE pull via AkShare (`--live`) when running in an online environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def generate_calibrated_datasets(seed: int = 202610) -> dict[str, Path]:
    """Generate realistic daily OHLCV/OI, macro factors, and AU2702 order-book snapshots."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Trading calendar spanning 2026-07-01 to 2026-11-20 (pre-warm indicators + Oct-Nov training window)
    dates = pd.bdate_range("2026-07-01", "2026-11-20")
    n = len(dates)

    # --- 1. Macro Drivers: US 10Y TIPS Real Yield (%) & US Dollar Index (DXY) ---
    tips = np.empty(n)
    dxy = np.empty(n)
    tips[0] = 2.05
    dxy[0] = 104.2
    for t in range(1, n):
        # Gradual dovish rate-expectation shift starting late Sep 2026
        drift_tips = -0.0045 if t > 55 else 0.001
        tips[t] = np.clip(tips[t - 1] + drift_tips + rng.normal(0, 0.028), 1.35, 2.45)
        drift_dxy = 1.6 * (tips[t] - tips[t - 1]) + rng.normal(0, 0.22)
        dxy[t] = np.clip(dxy[t - 1] + drift_dxy, 98.5, 108.5)

    macro_df = pd.DataFrame(
        {
            "date": dates,
            "us10y_tips_pct": np.round(tips, 4),
            "dxy_index": np.round(dxy, 3),
        }
    )

    # --- 2. SHFE Cast Aluminum Alloy AD2612 (Near) & AD2701 (Next) ---
    # Contract unit: 10 tons/lot, tick size: 5 RMB/ton
    ad2612_close = np.empty(n)
    ad2612_close[0] = 19850.0
    for t in range(1, n):
        # Cyclical industrial demand + mild mean reversion
        cycle = 18.0 * np.sin(2 * np.pi * t / 28.0)
        shock = rng.normal(0, 115.0)
        ad2612_close[t] = ad2612_close[t - 1] + 0.08 * (20150.0 - ad2612_close[t - 1]) + cycle + shock
    ad2612_close = np.round(ad2612_close / 5.0) * 5.0

    # Calendar spread (AD2612 - AD2701): mean-reverting contango/backwardation with inventory squeezes
    spread = np.empty(n)
    spread[0] = 45.0
    for t in range(1, n):
        # Inventory tightening spike around mid-Oct, then easing during rollover
        regime_mean = 95.0 if 62 <= t <= 78 else 25.0
        spread[t] = spread[t - 1] + 0.22 * (regime_mean - spread[t - 1]) + rng.normal(0, 22.0)
    spread = np.round(spread / 5.0) * 5.0
    ad2701_close = ad2612_close - spread

    # Liquidity & Open Interest migration from AD2612 -> AD2701 in early Nov 2026
    roll_progress = 1.0 / (1.0 + np.exp(-(np.arange(n) - 82) / 5.0))
    total_oi = 42000 + 6500 * np.sin(np.arange(n) / 14.0) + rng.normal(0, 1200, n)
    ad2612_oi = np.maximum(3500, (total_oi * (1 - 0.78 * roll_progress)).astype(int))
    ad2701_oi = np.maximum(4000, (total_oi * (0.22 + 0.78 * roll_progress)).astype(int))
    ad2612_vol = np.maximum(1500, (ad2612_oi * rng.uniform(0.55, 1.15, n)).astype(int))
    ad2701_vol = np.maximum(1800, (ad2701_oi * rng.uniform(0.50, 1.25, n)).astype(int))

    def _build_ohlc(close: np.ndarray, tick: float, vol_scale: float) -> pd.DataFrame:
        opens = np.roll(close, 1)
        opens[0] = close[0]
        opens = opens + np.round(rng.normal(0, vol_scale * 0.35, len(close)) / tick) * tick
        high_pad = np.round(np.abs(rng.normal(vol_scale * 0.7, vol_scale * 0.3, len(close))) / tick) * tick
        low_pad = np.round(np.abs(rng.normal(vol_scale * 0.7, vol_scale * 0.3, len(close))) / tick) * tick
        highs = np.maximum(opens, close) + high_pad
        lows = np.minimum(opens, close) - low_pad
        return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": close})

    ad2612_df = pd.concat(
        [
            pd.DataFrame({"date": dates, "contract": "AD2612"}),
            _build_ohlc(ad2612_close, tick=5.0, vol_scale=110.0),
            pd.DataFrame({"volume": ad2612_vol, "open_interest": ad2612_oi}),
        ],
        axis=1,
    )
    ad2701_df = pd.concat(
        [
            pd.DataFrame({"date": dates, "contract": "AD2701"}),
            _build_ohlc(ad2701_close, tick=5.0, vol_scale=105.0),
            pd.DataFrame({"volume": ad2701_vol, "open_interest": ad2701_oi}),
        ],
        axis=1,
    )

    # --- 3. SHFE Gold AU2702 (1,000 g/lot, tick size: 0.02 RMB/g) ---
    au_close = np.empty(n)
    au_close[0] = 612.40
    for t in range(1, n):
        # Gold inversely correlated with TIPS real yield & DXY, plus trend momentum
        dtips = tips[t] - tips[t - 1]
        ddxy = dxy[t] - dxy[t - 1]
        macro_push = -28.0 * dtips - 1.15 * ddxy
        trend_bias = 0.42 if t > 58 else 0.05
        au_close[t] = au_close[t - 1] + macro_push + trend_bias + rng.normal(0, 3.1)
    au_close = np.round(au_close / 0.02) * 0.02
    au_ohlc = _build_ohlc(au_close, tick=0.02, vol_scale=4.2)
    au_oi = (145000 + np.linspace(0, 48000, n) + rng.normal(0, 4500, n)).astype(int)
    au_vol = (au_oi * rng.uniform(0.75, 1.45, n)).astype(int)

    au2702_df = pd.concat(
        [
            pd.DataFrame({"date": dates, "contract": "AU2702"}),
            au_ohlc,
            pd.DataFrame({"volume": au_vol, "open_interest": au_oi}),
            macro_df[["us10y_tips_pct", "dxy_index"]],
        ],
        axis=1,
    )

    # --- 4. AU2702 Intraday 1-Minute Order-Book Microstructure Snapshot (Latest Session) ---
    minutes = pd.date_range("2026-11-20 09:00:00", "2026-11-20 11:30:00", freq="1min")
    m = len(minutes)
    mid = au_close[-1] + np.cumsum(rng.normal(0.015, 0.18, m))
    mid = np.round(mid / 0.02) * 0.02
    half_spread = rng.choice([0.02, 0.04, 0.06], size=m, p=[0.65, 0.25, 0.10])
    bid1 = np.round((mid - half_spread) / 0.02) * 0.02
    ask1 = np.round((mid + half_spread) / 0.02) * 0.02
    bid_vol_5 = rng.integers(35, 220, size=m)
    ask_vol_5 = rng.integers(35, 210, size=m)
    trade_vol = rng.integers(20, 380, size=m)
    ob_df = pd.DataFrame(
        {
            "timestamp": minutes,
            "contract": "AU2702",
            "bid1_px": bid1,
            "ask1_px": ask1,
            "mid_px": np.round((bid1 + ask1) / 2, 2),
            "bid_depth_5lvl": bid_vol_5,
            "ask_depth_5lvl": ask_vol_5,
            "trade_volume": trade_vol,
        }
    )

    paths = {
        "ad2612": DATA_DIR / "ad2612_daily.csv",
        "ad2701": DATA_DIR / "ad2701_daily.csv",
        "au2702": DATA_DIR / "au2702_daily.csv",
        "au2702_orderbook": DATA_DIR / "au2702_orderbook_1m.csv",
    }
    ad2612_df.to_csv(paths["ad2612"], index=False)
    ad2701_df.to_csv(paths["ad2701"], index=False)
    au2702_df.to_csv(paths["au2702"], index=False)
    ob_df.to_csv(paths["au2702_orderbook"], index=False)
    return paths


def fetch_live_shfe_akshare(symbol: str = "AU0") -> pd.DataFrame:
    """Optional live SHFE daily OHLCV fetcher using AkShare."""
    import akshare as ak  # type: ignore

    df = ak.futures_main_sina(symbol=symbol)
    df = df.rename(
        columns={
            "日期": "date",
            "开盘价": "open",
            "最高价": "high",
            "最低价": "low",
            "收盘价": "close",
            "成交量": "volume",
            "持仓量": "open_interest",
        }
    )
    return df[["date", "open", "high", "low", "close", "volume", "open_interest"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="SHFE Futures Data Pipeline")
    parser.add_argument("--live", action="store_true", help="Attempt live AkShare download for SHFE Gold")
    args = parser.parse_args()

    if args.live:
        try:
            live_df = fetch_live_shfe_akshare("AU0")
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            out = DATA_DIR / "au0_live_shfe.csv"
            live_df.to_csv(out, index=False)
            print(f"[LIVE] Saved {len(live_df)} rows to {out}")
            return
        except Exception as exc:
            print(f"[WARN] Live AkShare fetch failed ({exc}); falling back to calibrated dataset.")

    paths = generate_calibrated_datasets()
    for name, path in paths.items():
        print(f"[OK] Generated {name}: {path}")


if __name__ == "__main__":
    main()
