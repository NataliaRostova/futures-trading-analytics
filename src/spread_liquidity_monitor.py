"""Cast Aluminum Alloy (AD2612 / AD2701) Calendar Spread & Liquidity Monitor.

Tracks main vs. sub-main contract calendar spread, rolling Z-score mean-reversion bands,
Amihud illiquidity proxy, volume/OI turnover, and contract rollover timing.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"


def analyze_ad_spread_and_liquidity() -> dict:
    ad2612 = pd.read_csv(DATA_DIR / "ad2612_daily.csv", parse_dates=["date"])
    ad2701 = pd.read_csv(DATA_DIR / "ad2701_daily.csv", parse_dates=["date"])

    df = pd.merge(
        ad2612[["date", "close", "volume", "open_interest"]],
        ad2701[["date", "close", "volume", "open_interest"]],
        on="date",
        suffixes=("_2612", "_2701"),
    )

    df["spread"] = df["close_2612"] - df["close_2701"]
    df["spread_ma20"] = df["spread"].rolling(20).mean()
    df["spread_std20"] = df["spread"].rolling(20).std()
    df["spread_z"] = (df["spread"] - df["spread_ma20"]) / df["spread_std20"]

    # Annualized roll yield proxy (assuming ~30 days between Dec and Jan contracts)
    df["roll_yield_ann_pct"] = (df["spread"] / df["close_2701"]) * 12.0 * 100.0

    # Liquidity indicators
    df["oi_share_2701"] = df["open_interest_2701"] / (df["open_interest_2612"] + df["open_interest_2701"])
    df["vol_ratio_2701_2612"] = df["volume_2701"] / df["volume_2612"].clip(lower=1)

    # Amihud illiquidity proxy: |return| / notional turnover (in 100M RMB)
    ret_2612 = df["close_2612"].pct_change().abs()
    notional_2612_100m = (df["close_2612"] * df["volume_2612"] * 10.0) / 1e8
    df["amihud_2612"] = ret_2612 / notional_2612_100m.clip(lower=0.01)

    ret_2701 = df["close_2701"].pct_change().abs()
    notional_2701_100m = (df["close_2701"] * df["volume_2701"] * 10.0) / 1e8
    df["amihud_2701"] = ret_2701 / notional_2701_100m.clip(lower=0.01)

    # Identify main-contract rollover crossover date (OI share of AD2701 > 50%)
    rollover_rows = df[df["oi_share_2701"] > 0.50]
    rollover_date = rollover_rows["date"].iloc[0].strftime("%Y-%m-%d") if not rollover_rows.empty else "N/A"

    # Focus chart on training window (Sep–Nov 2026)
    view = df[df["date"] >= "2026-09-01"].copy()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.1, 1.0]})

    # Panel 1: Calendar Spread & Rolling Bands
    ax0 = axes[0]
    ax0.plot(view["date"], view["spread"], color="#1f4e79", lw=2.0, label="Calendar Spread (AD2612 - AD2701)")
    ax0.plot(view["date"], view["spread_ma20"], color="#e67e22", lw=1.4, ls="--", label="20D Rolling Mean")
    ax0.fill_between(
        view["date"],
        view["spread_ma20"] - 1.5 * view["spread_std20"],
        view["spread_ma20"] + 1.5 * view["spread_std20"],
        color="#3498db",
        alpha=0.14,
        label="±1.5σ Band",
    )
    ax0.axhline(0, color="#7f8c8d", lw=0.9, ls=":")
    ax0.set_ylabel("Spread (RMB/ton)")
    ax0.set_title("SHFE Cast Aluminum Alloy (AD2612 vs AD2701): Calendar Spread & Liquidity Migration", fontsize=13, fontweight="bold")
    ax0.legend(loc="upper left", fontsize=9)

    # Panel 2: Open Interest Migration
    ax1 = axes[1]
    ax1.plot(view["date"], view["open_interest_2612"] / 1000.0, color="#c0392b", lw=2.0, label="AD2612 Open Interest (k lots)")
    ax1.plot(view["date"], view["open_interest_2701"] / 1000.0, color="#27ae60", lw=2.0, label="AD2701 Open Interest (k lots)")
    if rollover_date != "N/A":
        rd = pd.Timestamp(rollover_date)
        ax1.axvline(rd, color="#8e44ad", ls="--", lw=1.4, label=f"Main Contract Rollover ({rollover_date})")
    ax1.set_ylabel("Open Interest (k lots)")
    ax1.legend(loc="upper left", fontsize=9)

    # Panel 3: Liquidity Volume Ratio & Z-Score
    ax2 = axes[2]
    ax2.bar(view["date"], view["vol_ratio_2701_2612"], color="#2980b9", alpha=0.55, width=0.8, label="Volume Ratio (AD2701 / AD2612)")
    ax2.axhline(1.0, color="#c0392b", lw=1.1, ls="--", label="Parity Threshold (1.0x)")
    ax2.set_ylabel("Volume Ratio")
    ax2.set_xlabel("Trading Date (2026)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    plt.tight_layout()
    chart_path = ASSETS_DIR / "ad_spread_liquidity.png"
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    oct_nov = df[df["date"] >= "2026-10-01"]
    summary = {
        "mean_spread_rmb": round(float(oct_nov["spread"].mean()), 2),
        "max_spread_rmb": round(float(oct_nov["spread"].max()), 2),
        "min_spread_rmb": round(float(oct_nov["spread"].min()), 2),
        "avg_roll_yield_ann_pct": round(float(oct_nov["roll_yield_ann_pct"].mean()), 2),
        "rollover_date": rollover_date,
        "latest_oi_share_2701_pct": round(float(df["oi_share_2701"].iloc[-1] * 100), 1),
        "avg_amihud_2612": round(float(oct_nov["amihud_2612"].mean()), 4),
        "avg_amihud_2701": round(float(oct_nov["amihud_2701"].mean()), 4),
        "chart_path": str(chart_path),
    }
    return summary


if __name__ == "__main__":
    res = analyze_ad_spread_and_liquidity()
    print("[AD Spread & Liquidity Summary]", res)
