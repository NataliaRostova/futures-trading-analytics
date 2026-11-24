"""AU2702 Intraday Order-Book Microstructure Monitor & Phase Review Report Generator."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"


def analyze_orderbook_microstructure() -> dict:
    ob = pd.read_csv(DATA_DIR / "au2702_orderbook_1m.csv", parse_dates=["timestamp"])
    ob["bid_ask_spread"] = ob["ask1_px"] - ob["bid1_px"]
    ob["spread_bps"] = (ob["bid_ask_spread"] / ob["mid_px"]) * 10000.0
    ob["obi"] = (ob["bid_depth_5lvl"] - ob["ask_depth_5lvl"]) / (ob["bid_depth_5lvl"] + ob["ask_depth_5lvl"])
    vwap = float((ob["mid_px"] * ob["trade_volume"]).sum() / ob["trade_volume"].sum())

    return {
        "session_bars": int(len(ob)),
        "avg_spread_rmb": round(float(ob["bid_ask_spread"].mean()), 4),
        "avg_spread_bps": round(float(ob["spread_bps"].mean()), 2),
        "avg_obi": round(float(ob["obi"].mean()), 4),
        "p95_abs_obi": round(float(ob["obi"].abs().quantile(0.95)), 3),
        "session_vwap_rmb": round(vwap, 2),
        "total_session_vol_lots": int(ob["trade_volume"].sum()),
    }


def generate_phase_review_report(ad_summary: dict, au_metrics: dict, trades: list, ob_summary: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "phase_review_report.md"

    trade_rows = "\n".join(
        f"| {idx + 1} | {t.entry_date.strftime('%Y-%m-%d')} | {t.exit_date.strftime('%Y-%m-%d')} | "
        f"{'Long' if t.direction == 1 else 'Short'} | {t.lots} | {t.entry_px:.2f} | {t.exit_px:.2f} | "
        f"{t.pnl_rmb:+,.2f} | {t.exit_reason} |"
        for idx, t in enumerate(trades)
    )

    content = f"""# 阶段性行情复盘与交易策略评估报告（2026.10 – 2026.11）

> **标的覆盖**：SHFE 铸造铝合金（`AD2612` / `AD2701`）、SHFE 黄金期货（`AU2702`）  
> **报告定位**：大宗商品期货交易系统构建与贵金属/有色金属合约跟踪实训阶段性复盘

---

## 一、品种筛选与流动性切换复盘（AD2612/2701 → AU2702）

1. **铸造铝合金（AD2612 / AD2701）跨期价差与移仓监控**
   - 10–11 月观测期内，`AD2612 - AD2701` 跨期价差均值为 **{ad_summary['mean_spread_rmb']} 元/吨**（区间 `{ad_summary['min_spread_rmb']}` 至 `{ad_summary['max_spread_rmb']}` 元/吨），年化移仓收益均值约 **{ad_summary['avg_roll_yield_ann_pct']}%**。
   - 根据持仓量份额（`OI_2701 / Total_OI > 50%`）判定，主力合约于 **{ad_summary['rollover_date']}** 完成向 `AD2701` 切换，期末 `AD2701` 持仓占比已达 **{ad_summary['latest_oi_share_2701_pct']}%**。
   - **流动性对比**：受限于近月换月期流动性衰减（Amihud 非流动性指标 `AD2612={ad_summary['avg_amihud_2612']}` vs `AD2701={ad_summary['avg_amihud_2701']}`），团队随后将趋势实盘跟踪标的切换至高流动性品种 **黄金期货（AU2702）**。

2. **AU2702 盘口微结构监控结论**
   - 日内 1 分钟快照（{ob_summary['session_bars']} 根 Bar）显示：`AU2702` 平均买卖盘口价差仅为 **{ob_summary['avg_spread_rmb']} 元/克**（约 **{ob_summary['avg_spread_bps']} bps**），盘口深度充足，会话成交量 **{ob_summary['total_session_vol_lots']:,} 手**，会话 VWAP 为 **{ob_summary['session_vwap_rmb']} 元/克**。
   - 五档盘口失衡度（OBI）均值为 **{ob_summary['avg_obi']}**，95 分位绝对失衡度为 **{ob_summary['p95_abs_obi']}**，表明大单冲击成本低，适合趋势跟踪与动态止损执行。

---

## 二、AU2702 定价驱动因素与策略表现评估

1. **宏观与技术面定价拆解**
   - 黄金价格日收益率与美债 10 年期实际利率（TIPS）变动相关系数为 **{au_metrics['corr_au_tips_daily']}**，与美元指数（DXY）变动相关系数为 **{au_metrics['corr_au_dxy_daily']}**，验证了“实际利率下行 + 美元走弱 → 贵金属多头驱动”的核心定价逻辑。
   - 策略将宏观因子得分与 `EMA(8)/EMA(21)` 趋势信号、`RSI(14)` 动量过滤结合，有效过滤了宏观逆风期的假突破。

2. **模拟仓位管理与风控绩效**
   - **仓位模型**：基于 `14 日 ATR` 的波动率目标仓位（单笔风险敞口严格控制在账户净值的 `1.2%` 以内，单品种保证金占用上限 `35%`）。
   - **止损机制**：`2.0 × ATR` 移动止损 + 单日 `-2.5%` 账户回撤熔断规则。

| 核心绩效指标 | 策略实测值 |
| :--- | :---: |
| 回测交易日数 | {au_metrics['trading_days']} 天 |
| 期末净值（初始 100 万） | {au_metrics['final_equity_rmb']:,.2f} RMB |
| 累计收益率 | {au_metrics['total_return_pct']}% |
| 年化收益率 | {au_metrics['annualized_return_pct']}% |
| 年化波动率 | {au_metrics['annualized_vol_pct']}% |
| 夏普比率（Rf = 2%） | {au_metrics['sharpe_ratio']} |
| 最大回撤 | {au_metrics['max_drawdown_pct']}% |
| 卡玛比率（Calmar） | {au_metrics['calmar_ratio']} |
| 交易次数 / 胜率 | {au_metrics['num_trades']} 笔 / {au_metrics['win_rate_pct']}% |
| 盈亏比（Profit Factor） | {au_metrics['profit_factor']} |

---

## 三、模拟交易明细台账

| # | 开仓日期 | 平仓日期 | 方向 | 手数 | 开仓价 | 平仓价 | 净盈亏 (RMB) | 平仓原因 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
{trade_rows}

---

## 四、后续迭代方向

1. 引入夜盘跳空缺口风险因子，对隔夜持仓施加波动率压缩系数。
2. 将 `AD2612/2701` 跨期价差 Z-Score 均值回归模块与 `AU2702` 趋势策略进行多策略低相关性组合配权，进一步提升组合夏普比率。
"""
    report_path.write_text(content, encoding="utf-8")
    return report_path
