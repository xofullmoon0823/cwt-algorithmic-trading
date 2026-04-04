# ============================================================
# 그림 수정 스크립트
# Fig1, Fig2, Fig4, Fig7, Fig3_equity_curves 수정
# ============================================================

import os, warnings
import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import platform

warnings.filterwarnings("ignore")

if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

TICKERS     = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]
TEST_START  = "2023-01-01"
TEST_END    = "2025-12-31"
INIT_CASH   = 100_000.0
TC          = 0.001
CWT_SCALES  = np.arange(1, 65)
CWT_WAVELET = "morl"
CWT_A_TH    = 24
ATH_CANDIDATES = [4, 8, 12, 16, 24, 32]

def cwt_clean(close_vals, a_th, wavelet=CWT_WAVELET):
    mu, sigma = close_vals.mean(), close_vals.std() or 1.0
    x_norm = (close_vals - mu) / sigma
    coef, _ = pywt.cwt(x_norm, CWT_SCALES, wavelet)
    coef_c  = coef.copy()
    coef_c[CWT_SCALES < a_th, :] = 0
    n_low   = (CWT_SCALES >= a_th).sum()
    x_c     = np.sum(coef_c, axis=0) / n_low if n_low > 0 else x_norm
    clean   = x_c * sigma + mu
    noise   = close_vals - clean
    noise_p = np.mean(noise**2)
    sig_p   = np.mean(clean**2) + 1e-10
    snr     = 10 * np.log10(sig_p / (noise_p + 1e-10))
    noise_r = noise_p / (sig_p + noise_p)
    return clean, noise, snr, noise_r

print("데이터 로드 중...")
price_data = {}
for ticker in TICKERS:
    df = pd.read_csv(f"data/price/{ticker}_price_v4.csv",
                     index_col="Date", parse_dates=True)
    price_data[ticker] = df

aapl_close = price_data["AAPL"]["Close"].values
aapl_clean, aapl_noise, _, _ = cwt_clean(aapl_close, CWT_A_TH)
aapl_clean_s = pd.Series(aapl_clean, index=price_data["AAPL"].index)
aapl_noise_s = pd.Series(aapl_noise, index=price_data["AAPL"].index)

abl_df  = pd.read_csv("results/ablation/ablation_ath_v4.csv")
abl_avg = abl_df.groupby("a_th")[["ARR(%)","SR","MDD(%)","SNR(dB)","Noise(%)"]].mean().round(3)
print("데이터 로드 완료\n")

# ── Fig1 ────────────────────────────────────────────────
print("[Fig1] 범례 a_th 수정...")
td = price_data["AAPL"].loc[TEST_START:TEST_END].copy()
td["Close_clean"] = aapl_clean_s.loc[TEST_START:TEST_END]
td["Noise"]       = aapl_noise_s.loc[TEST_START:TEST_END]

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
axes[0].plot(td.index, td["Close"], color="#AAAAAA", lw=0.9, label="원 신호 (Raw)")
axes[0].plot(td.index, td["Close_clean"], color="#2E75B6", lw=1.8,
             label=f"CWT 클린 신호 (a_th={CWT_A_TH})")
axes[0].set_title("AAPL — CWT 필터 전/후 비교 (테스트 기간)", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Price (USD)"); axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3)
axes[1].fill_between(td.index, td["Noise"], 0, where=td["Noise"]>=0,
                      color="#C55A11", alpha=0.65, label="양(+) 노이즈")
axes[1].fill_between(td.index, td["Noise"], 0, where=td["Noise"]<0,
                      color="#1E6B3C", alpha=0.65, label="음(-) 노이즈")
axes[1].axhline(0, color="black", lw=0.7)
axes[1].set_title("제거된 노이즈 성분", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Noise (USD)"); axes[1].set_xlabel("Date")
axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/cwt/Fig1_cwt_before_after_AAPL.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Fig1 저장\n")

# ── Fig2 ────────────────────────────────────────────────
print("[Fig2] 제목 수정 + a_th=24,32 범례 추가...")
close_all = price_data["AAPL"]["Close"].values
dates_t   = price_data["AAPL"].loc[TEST_START:TEST_END].index
ath_colors = ["#AAAAAA", "#4472C4", "#1E6B3C", "#7030A0", "#2E75B6", "#C55A11"]

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(dates_t, price_data["AAPL"].loc[TEST_START:TEST_END]["Close"].values,
        color="#DDDDDD", lw=0.8, alpha=0.7, label="원 신호 (Raw)")
for a_th, col in zip(ATH_CANDIDATES, ath_colors):
    clean, _, snr, nr = cwt_clean(close_all, a_th)
    cs = pd.Series(clean, index=price_data["AAPL"].index)
    ct = cs.loc[TEST_START:TEST_END]
    row = abl_df[(abl_df["Ticker"]=="AAPL") & (abl_df["a_th"]==a_th)]
    arr_txt = f"{row['ARR(%)'].values[0]:.1f}%" if len(row) else ""
    lw_val = 2.5 if a_th == CWT_A_TH else 1.2
    ls_val = "-"  if a_th == CWT_A_TH else "--"
    marker = " ★최적" if a_th == CWT_A_TH else ""
    ax.plot(ct.index, ct.values, color=col, lw=lw_val, linestyle=ls_val,
            label=f"a_th={a_th}{marker}  노이즈 {nr*100:.0f}% 제거  ARR={arr_txt}")
ax.set_title(f"Ablation Study — 스케일 임계값(a_th) 비교 (AAPL)\n최적 a_th={CWT_A_TH}",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/cwt/Fig2_ablation_threshold_AAPL.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Fig2 저장\n")

# ── Fig4 ────────────────────────────────────────────────
print("[Fig4] SNR/노이즈 수치 실험값으로 수정...")
snr_vals   = {"AAPL":15.6, "AMZN":12.2, "GOOGL":13.6, "MSFT":13.3, "TSLA":12.7}
noise_vals = {"AAPL": 2.7, "AMZN": 5.7, "GOOGL": 4.2, "MSFT": 4.5, "TSLA": 5.1}
snr_v   = [snr_vals[t]   for t in TICKERS]
noise_v = [noise_vals[t] for t in TICKERS]
bc = ["#2E75B6"]*4 + ["#C55A11"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
b1 = ax1.bar(TICKERS, snr_v, color=bc, alpha=0.85, width=0.5)
for bar, val in zip(b1, snr_v):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
             f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
ax1.set_title("SNR (dB) by Ticker\n(높을수록 클린 신호)", fontsize=12, fontweight="bold")
ax1.set_ylabel("SNR (dB)"); ax1.set_ylim(0, 20); ax1.grid(True, alpha=0.3, axis="y")

b2 = ax2.bar(TICKERS, noise_v, color=bc, alpha=0.85, width=0.5)
for bar, val in zip(b2, noise_v):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
             f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax2.set_title("Noise Ratio (%) by Ticker\n(낮을수록 클린 신호)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Noise Ratio (%)"); ax2.set_ylim(0, 8); ax2.grid(True, alpha=0.3, axis="y")
ax2.annotate("TSLA\n고변동성", xy=(4, noise_vals["TSLA"]), xytext=(3.0, 6.8),
             fontsize=9, color="#C55A11", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#C55A11"))
plt.suptitle("Fig 4. 종목별 CWT 노이즈 특성", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/cwt/Fig4_snr_by_ticker.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Fig4 저장\n")

# ── Fig7 ────────────────────────────────────────────────
print("[Fig7] 노란 테두리 a_th=24로 이동, 제목 수정...")
abl_avg_plot = abl_avg.reset_index()
best_idx = list(abl_avg_plot["a_th"].values).index(CWT_A_TH)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, metric, title, col in zip(
    axes,
    ["ARR(%)", "SR", "MDD(%)"],
    ["ARR (%) — 연간 수익률", "SR — 샤프 지수", "MDD (%) — 최대 낙폭"],
    ["#2E75B6", "#1E6B3C", "#C55A11"]
):
    vals = abl_avg_plot[metric].values
    athl = [f"a_th={int(v)}" for v in abl_avg_plot["a_th"].values]
    bars = ax.bar(athl, vals, color=col, alpha=0.8, width=0.5)
    bars[best_idx].set_edgecolor("#FFD700")
    bars[best_idx].set_linewidth(3)
    for bar, val in zip(bars, vals):
        off = abs(val)*0.03+0.03
        ax.text(bar.get_x()+bar.get_width()/2, val+(off if val>=0 else -off),
                f"{val:.2f}", ha="center",
                va="bottom" if val>=0 else "top", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(metric)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", labelsize=8)
plt.suptitle(
    f"Fig 7. Ablation Study — a_th별 성능 비교 (5개 종목 평균)\n"
    f"노란 테두리: 최적 a_th={CWT_A_TH}",
    fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/Fig7_ablation_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Fig7 저장\n")

# ── Fig3_equity_curves ──────────────────────────────────
print("[Fig3_equity_curves] 2023-01~2025-12 기간으로 재생성...")
eq_paths = {
    "B&H":            "results/baselines/AAPL_B&H_equity.csv",
    "MACD":           "results/baselines/AAPL_MACD_equity.csv",
    "RSI":            "results/baselines/AAPL_RSI_equity.csv",
    "SMA":            "results/baselines/AAPL_SMA_equity.csv",
    "FinAgent(원본)": "results/agent/AAPL_FinAgent원본_equity.csv",
    "CWT-FinAgent":   "results/agent/AAPL_CWT-FinAgent제안_equity.csv",
}
eq_style = {
    "B&H":            ("#888888", "--", 1.2),
    "MACD":           ("#C55A11", "--", 1.2),
    "RSI":            ("#7030A0", "--", 1.2),
    "SMA":            ("#4472C4", "--", 1.2),
    "FinAgent(원본)": ("#1E6B3C", "--", 1.4),
    "CWT-FinAgent":   ("#2E75B6", "-",  2.5),
}
fig, ax = plt.subplots(figsize=(13, 6))
loaded = 0
for label, path in eq_paths.items():
    if os.path.exists(path):
        eq = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        eq = eq.loc[TEST_START:TEST_END]
        col, ls, lw = eq_style[label]
        ax.plot(eq.index, eq.values, color=col, lw=lw, linestyle=ls, label=label)
        loaded += 1
    else:
        print(f"  ⚠️  {path} 없음")
ax.axhline(y=INIT_CASH, color="black", lw=0.8, linestyle=":", alpha=0.4, label="초기 자금")
ax.set_title(
    f"AAPL — 전략별 에쿼티 커브 비교\n(테스트: {TEST_START}~{TEST_END}, 거래비용 {TC*100:.1f}%)",
    fontsize=13, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Portfolio Value (USD)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/paper_tables/Fig3_equity_curves_AAPL.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Fig3_equity_curves 저장 ({loaded}개 전략)\n")

print("="*50)
print("  모든 그림 수정 완료!")
print("  results/cwt/Fig1_cwt_before_after_AAPL.png")
print("  results/cwt/Fig2_ablation_threshold_AAPL.png")
print("  results/cwt/Fig4_snr_by_ticker.png")
print("  results/figures/Fig7_ablation_performance.png")
print("  results/paper_tables/Fig3_equity_curves_AAPL.png")
print("="*50)
