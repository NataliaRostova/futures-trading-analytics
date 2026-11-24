"""End-to-end runner for the SHFE Futures Trading & Commodity Analytics Framework."""

from __future__ import annotations

import json
from pathlib import Path

from src.data_pipeline import generate_calibrated_datasets
from src.spread_liquidity_monitor import analyze_ad_spread_and_liquidity
from src.gold_strategy_engine import run_gold_backtest
from src.risk_and_report import analyze_orderbook_microstructure, generate_phase_review_report


def main() -> None:
    print("1/4 Generating/loading SHFE futures & macro datasets...")
    generate_calibrated_datasets()

    print("2/4 Running Cast Aluminum Alloy (AD2612/AD2701) spread & liquidity monitor...")
    ad_summary = analyze_ad_spread_and_liquidity()

    print("3/4 Running Gold Futures (AU2702) macro-technical strategy & risk engine...")
    _, trades, au_metrics = run_gold_backtest()

    print("4/4 Analyzing AU2702 order-book microstructure & compiling review report...")
    ob_summary = analyze_orderbook_microstructure()
    report_path = generate_phase_review_report(ad_summary, au_metrics, trades, ob_summary)

    summary_out = Path(__file__).resolve().parent / "reports" / "metrics_summary.json"
    summary_out.write_text(
        json.dumps({"ad_spread": ad_summary, "au_strategy": au_metrics, "au_orderbook": ob_summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[DONE] Report generated at: {report_path}")
    print(json.dumps({"ad_spread": ad_summary, "au_strategy": au_metrics, "au_orderbook": ob_summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
