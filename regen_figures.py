#!/usr/bin/env python3
# ============================================================
# Fig3 / Fig5 / Fig6 / Fig8 재생성
#   "FinAgent(원본)"    → "MA 기준전략(미적용)"
#   "CWT-FinAgent(제안)" → "CWT-MA 전략(제안)"
#   "CWT-FinAgent"      → "CWT-MA 전략"
# 실행: python3 regen_figures.py
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import platform

# ── 한글 폰트 ─────────────────────────────────────────────
if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
elif platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

OLD_AGENT = "FinAgent(원본)"
NEW_AGENT = "MA 기준전략(미적용)"
OLD_CWT   = "CWT-FinAgent(제안)"
NEW_CWT   = "CWT-MA 전략(제안)"

TICKERS          = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]
INIT_CASH        = 100_000.0
TRANSACTION_COST = 0.001
TEST_START       = "2023-01-01"
TEST_END         = "2025-12-31"

strategies_order = ["B&H", "MACD", "RSI", "SMA", NEW_AGENT, NEW_CWT]
color_map = {
    "B&H":       "#888888",
    "MACD":      "#C55A11",
    "RSI":       "#7030A0",
    "SMA":       "#4472C4",
    NEW_AGENT:   "#1E6B3C",
    NEW_CWT:     "#2E75B6",
}

# ── CSV 로드 & 레이블 정규화 ───────────────────────────────
def load_all_results() -> pd.DataFrame:
    df = pd.read_csv("results/paper_tables/ALL_results_v4.csv")
    df["Strategy"] = (df["Strategy"]
                      .str.replace(OLD_AGENT, NEW_AGENT, regex=False)
                      .str.replace(OLD_CWT,   NEW_CWT,   regex=False))
    return df

def load_stat_records() -> list:
    df = pd.read_csv("results/stats/statistical_tests_v4.csv")
    return df.to_dict("records")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig3: 성능 비교 막대 그래프
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def make_fig3(all_df: pd.DataFrame):
    avg_all = all_df.groupby("Strategy")[
        ["ARR(%)", "SR", "CR", "SOR", "MDD(%)", "VOL(%)", "Win Rate(%)"]
    ].mean().round(3)

    avg_plot = avg_all.reindex(
        [s for s in strategies_order if s in avg_all.index])
    labels = [
        s.replace("(제안)", "\n(제안)").replace("(미적용)", "\n(미적용)")
        for s in avg_plot.index
    ]
    colors = [color_map.get(s, "#888888") for s in avg_plot.index]

    fig, axes = plt.subplots(3, 1, figsize=(11, 13))
    for ax, metric, title, ylabel in zip(
        axes,
        ["ARR(%)", "SR", "MDD(%)"],
        ["ARR (%) — 연간 수익률", "SR — 샤프 지수", "MDD (%) — 최대 낙폭"],
        ["ARR (%)", "Sharpe Ratio", "MDD (%)"],
    ):
        vals = avg_plot[metric].values
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.5)
        cwt_idx = next(
            (i for i, s in enumerate(avg_plot.index) if "CWT" in s), -1)
        if cwt_idx >= 0:
            bars[cwt_idx].set_edgecolor("#FFD700")
            bars[cwt_idx].set_linewidth(3)
        for bar, val in zip(bars, vals):
            off = abs(val) * 0.04 + 0.05
            yp  = val + off if val >= 0 else val - off
            ax.text(bar.get_x() + bar.get_width() / 2, yp,
                    f"{val:.2f}", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8.5, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelsize=9)

    plt.suptitle(
        f"전략별 성능 비교 — 5개 종목 평균\n"
        f"(테스트: {TEST_START}~{TEST_END}, 거래비용 {TRANSACTION_COST*100:.1f}%)",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "results/paper_tables/Fig3_performance_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Fig3 저장: {out}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig5: AAPL 에쿼티 커브
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def make_fig5():
    eq_paths = {
        "B&H":       "results/baselines/AAPL_B&H_equity.csv",
        "MACD":      "results/baselines/AAPL_MACD_equity.csv",
        "RSI":       "results/baselines/AAPL_RSI_equity.csv",
        "SMA":       "results/baselines/AAPL_SMA_equity.csv",
        NEW_AGENT:   "results/agent/AAPL_FinAgent원본_equity.csv",
        "CWT-MA 전략": "results/agent/AAPL_CWT-FinAgent제안_equity.csv",
    }
    eq_style = {
        "B&H":         ("#888888", "--", 1.2),
        "MACD":        ("#C55A11", "--", 1.2),
        "RSI":         ("#7030A0", "--", 1.2),
        "SMA":         ("#4472C4", "--", 1.2),
        NEW_AGENT:     ("#1E6B3C", "--", 1.2),
        "CWT-MA 전략": ("#2E75B6", "-",  2.2),
    }
    fig, ax = plt.subplots(figsize=(13, 6))
    for label, path in eq_paths.items():
        if os.path.exists(path):
            eq = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
            col, ls, lw = eq_style[label]
            ax.plot(eq.index, eq.values,
                    color=col, lw=lw, linestyle=ls, label=label)
    ax.axhline(y=INIT_CASH, color="black", lw=0.8,
               linestyle=":", alpha=0.4, label="초기 자금")
    ax.set_title(
        f"AAPL — 전략별 에쿼티 커브 비교\n"
        f"(테스트: {TEST_START}~{TEST_END}, 거래비용 {TRANSACTION_COST*100:.1f}%)",
        fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = "results/paper_tables/Fig5_equity_curves_AAPL.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Fig5 저장: {out}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig6: 종목별 ARR 비교
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def make_fig6(all_df: pd.DataFrame):
    pivot = all_df[all_df["Strategy"].isin(
        [NEW_AGENT, NEW_CWT])].pivot(
        index="Ticker", columns="Strategy", values="ARR(%)")
    pivot["개선(CWT-원본)"] = pivot[NEW_CWT] - pivot[NEW_AGENT]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(TICKERS))
    w = 0.35

    axes[0].bar(x - w / 2, pivot[NEW_AGENT].values,
                w, label=NEW_AGENT, color="#1E6B3C", alpha=0.85)
    axes[0].bar(x + w / 2, pivot[NEW_CWT].values,
                w, label=NEW_CWT, color="#2E75B6",
                alpha=0.85, edgecolor="#FFD700", linewidth=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(TICKERS)
    axes[0].set_title("종목별 ARR 비교", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("ARR (%)")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")
    for i, (raw_v, cwt_v) in enumerate(
            zip(pivot[NEW_AGENT].values, pivot[NEW_CWT].values)):
        axes[0].text(i - w / 2, raw_v + (0.5 if raw_v >= 0 else -1.5),
                     f"{raw_v:.1f}%", ha="center", fontsize=8)
        axes[0].text(i + w / 2, cwt_v + (0.5 if cwt_v >= 0 else -1.5),
                     f"{cwt_v:.1f}%", ha="center", fontsize=8,
                     fontweight="bold")

    diff_v = pivot["개선(CWT-원본)"].values
    bc2    = ["#2E75B6" if v >= 0 else "#C55A11" for v in diff_v]
    bars2  = axes[1].bar(TICKERS, diff_v, color=bc2, alpha=0.85, width=0.5)
    for bar, val in zip(bars2, diff_v):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     val + (0.3 if val >= 0 else -0.8),
                     f"{val:+.1f}%p", ha="center", fontsize=9,
                     fontweight="bold")
    axes[1].axhline(0, color="black", lw=1.2)
    axes[1].set_title("CWT 전처리 ARR 개선량 (CWT − 원본)",
                       fontsize=12, fontweight="bold")
    axes[1].set_ylabel("ARR 개선량 (%p)")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Fig 6. 종목별 성능 비교 — CWT 전처리 효과",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "results/figures/Fig6_ticker_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Fig6 저장: {out}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig8: 통계 검정 결과 시각화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def make_fig8(stat_records: list):
    if not stat_records:
        print("  ⚠️  Fig8: stat_records 없음, 건너뜀")
        return

    tickers_s = [r["Ticker"]       for r in stat_records]
    p_vals    = [r["p-value"]       for r in stat_records]
    cwt_rets  = [r["CWT mean ret"]  for r in stat_records]
    raw_rets  = [r["RAW mean ret"]  for r in stat_records]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bar_colors = ["#2E75B6" if p < 0.05 else "#AAAAAA" for p in p_vals]
    axes[0].bar(tickers_s, [-np.log10(p) for p in p_vals],
                color=bar_colors, alpha=0.85, width=0.5)
    axes[0].axhline(-np.log10(0.05), color="red", lw=1.5,
                    linestyle="--", label="p=0.05 기준선")
    axes[0].axhline(-np.log10(0.01), color="darkred", lw=1.5,
                    linestyle=":", label="p=0.01 기준선")
    axes[0].set_title("-log10(p-value)\n(막대가 높을수록 통계적 유의)",
                       fontsize=11, fontweight="bold")
    axes[0].set_ylabel("-log10(p-value)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")

    x = np.arange(len(tickers_s))
    w = 0.35
    axes[1].bar(x - w / 2, raw_rets, w,
                label=NEW_AGENT, color="#1E6B3C", alpha=0.85)
    axes[1].bar(x + w / 2, cwt_rets, w,
                label="CWT-MA 전략", color="#2E75B6", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tickers_s)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("일평균 수익률 비교\n(CWT vs 원본)",
                       fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Daily Mean Return (%)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Fig 8. 통계적 유의성 검증 결과",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "results/figures/Fig8_statistical_tests.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Fig8 저장: {out}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 65)
print('  범례 변경:')
print('    "FinAgent(원본)"     → "MA 기준전략(미적용)"')
print('    "CWT-FinAgent(제안)" → "CWT-MA 전략(제안)"')
print('    "CWT-FinAgent"       → "CWT-MA 전략"')
print("  대상: Fig3, Fig5, Fig6, Fig8")
print("  확인(변경 없음): Fig1, Fig2, Fig7")
print("=" * 65)

all_df       = load_all_results()
stat_records = load_stat_records()

make_fig3(all_df)
make_fig5()
make_fig6(all_df)
make_fig8(stat_records)

print("\n  ℹ  Fig1 — 레이블: '원 신호 (Raw)', 'CWT 클린 신호' → 변경 불필요")
print("  ℹ  Fig2 — 레이블: 'a_th=X ...'                   → 변경 불필요")
print("  ℹ  Fig7 — 레이블: 'a_th=X ...'                   → 변경 불필요")
print("\n완료.")
