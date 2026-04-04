# ============================================================
# 그림만 다시 뽑기 (실험 없이 그림 3개만 생성)
# 변문성 (20223872)
# ============================================================
# 실행: py make_figures.py  (그램)
#       python make_figures.py  (맥미니)
# ============================================================

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pywt
warnings.filterwarnings("ignore")

# ── 한글 폰트 설정 ──────────────────────────────────────────
import platform
if platform.system() == "Darwin":
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == "Windows":
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 경로 설정 ───────────────────────────────────────────────
DATA_DIR    = "data/price"
CLEAN_DIR   = "data/clean"
AGENT_DIR   = "results/agent"
BASELINE_DIR= "results/baselines"
CWT_DIR     = "results/cwt"
TABLE_DIR   = "results/paper_tables"

TEST_START  = "2023-06-01"
TEST_END    = "2024-01-01"
INIT_CASH   = 100_000

os.makedirs(CWT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

print("=" * 50)
print("  논문용 그림 생성 시작")
print("=" * 50)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CWTFilter 클래스
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CWTFilter:
    def __init__(self, scale_threshold=8):
        self.scales = np.arange(1, 65)
        self.a_th   = scale_threshold

    def transform(self, close):
        mu, sigma = close.mean(), close.std() or 1.0
        x_norm    = (close - mu) / sigma
        coef, _   = pywt.cwt(x_norm, self.scales, 'morl')
        coef_c    = coef.copy()
        coef_c[self.scales < self.a_th, :] = 0
        n_low     = (self.scales >= self.a_th).sum()
        x_cn      = np.sum(coef_c, axis=0) / n_low if n_low > 0 else x_norm
        x_clean   = x_cn * sigma + mu
        x_noise   = close - x_clean
        noise_p   = np.mean(x_noise**2)
        sig_p     = np.mean(x_clean**2) + 1e-10
        return {
            "clean": x_clean, "noise": x_noise,
            "coef": coef, "coef_c": coef_c,
            "noise_ratio": noise_p / (sig_p + noise_p),
        }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig 1: CWT 필터 전/후 (논문 3장)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n  Fig1 생성 중...")

# 클린 신호 파일 로드 (없으면 직접 계산)
clean_path = f"{CLEAN_DIR}/AAPL_clean_ath8.csv"
price_path = f"{DATA_DIR}/AAPL_price.csv"

if os.path.exists(clean_path):
    df = pd.read_csv(clean_path, index_col="Date", parse_dates=True)
else:
    df = pd.read_csv(price_path, index_col="Date", parse_dates=True)
    cwt = CWTFilter(scale_threshold=8)
    res = cwt.transform(df["Close"].values)
    df["Close_clean"] = res["clean"]
    df["Noise"]       = res["noise"]

test = df.loc[TEST_START:TEST_END]

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

axes[0].plot(test.index, test["Close"],       color="#AAAAAA", lw=0.9, label="원 신호 (Raw)")
axes[0].plot(test.index, test["Close_clean"], color="#2E75B6", lw=1.6, label="CWT 클린 신호")
axes[0].set_title("AAPL — CWT 필터 전/후 가격 신호 비교 (테스트 기간)",
                  fontsize=13, fontweight="bold")
axes[0].set_ylabel("Price (USD)")
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].fill_between(test.index, test["Noise"], 0,
                     where=test["Noise"] >= 0, color="#C55A11", alpha=0.6, label="양(+) 노이즈")
axes[1].fill_between(test.index, test["Noise"], 0,
                     where=test["Noise"] < 0,  color="#1E6B3C", alpha=0.6, label="음(-) 노이즈")
axes[1].axhline(0, color="black", lw=0.7)
axes[1].set_title("제거된 노이즈 성분", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Noise (USD)")
axes[1].set_xlabel("Date")
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CWT_DIR}/Fig1_cwt_before_after_AAPL.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Fig1 저장: {CWT_DIR}/Fig1_cwt_before_after_AAPL.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig 2: Ablation Study (논문 5장)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n  Fig2 생성 중...")

df_price = pd.read_csv(price_path, index_col="Date", parse_dates=True)
dates_test = df_price.loc[TEST_START:TEST_END].index
close_all  = df_price["Close"].values

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(dates_test, df_price.loc[TEST_START:TEST_END]["Close"].values,
        color="#CCCCCC", lw=0.8, alpha=0.8, label="원 신호")

colors = ["#2E75B6", "#C55A11", "#1E6B3C", "#7030A0"]
for a_th, col in zip([4, 8, 16, 32], colors):
    cwt = CWTFilter(scale_threshold=a_th)
    res = cwt.transform(close_all)
    cs  = pd.Series(res["clean"], index=df_price.index)
    ct  = cs.loc[TEST_START:TEST_END]
    ax.plot(ct.index, ct.values, color=col, lw=1.4,
            label=f"a_th={a_th}  (노이즈 {res['noise_ratio']*100:.0f}% 제거)")

ax.set_title("Ablation Study — 스케일 임계값(a_th) 비교 (AAPL)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{CWT_DIR}/Fig2_ablation_threshold_AAPL.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Fig2 저장: {CWT_DIR}/Fig2_ablation_threshold_AAPL.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig 3: 에쿼티 커브 (논문 4장)
# 저장된 에쿼티 CSV 파일 사용
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n  Fig3 생성 중...")

fig, ax = plt.subplots(figsize=(13, 6))

# 로드할 파일 목록
eq_map = {
    "B&H":            (f"{BASELINE_DIR}/AAPL_B&H_equity.csv",         "#888888", "--", 1.3),
    "MACD":           (f"{BASELINE_DIR}/AAPL_MACD_equity.csv",        "#C55A11", "--", 1.3),
    "FinAgent(원본)": (f"{AGENT_DIR}/AAPL_FinAgent원본_equity.csv",   "#1E6B3C", "--", 1.3),
    "CWT-FinAgent":   (f"{AGENT_DIR}/AAPL_CWT-FinAgent제안_equity.csv","#2E75B6", "-",  2.0),
}

plotted = False
for label, (path, col, ls, lw) in eq_map.items():
    if os.path.exists(path):
        eq = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        ax.plot(eq.index, eq.values, color=col, lw=lw,
                linestyle=ls, label=label)
        plotted = True
    else:
        print(f"    ⚠️  파일 없음: {path}")

if plotted:
    ax.axhline(y=INIT_CASH, color="black", lw=0.8, linestyle=":", alpha=0.4, label="초기 자금")
    ax.set_title("AAPL — 전략별 에쿼티 커브 비교 (테스트: 2023.06~2024.01)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{TABLE_DIR}/Fig3_equity_curves_AAPL.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Fig3 저장: {TABLE_DIR}/Fig3_equity_curves_AAPL.png")
else:
    print("  ❌ Fig3: 에쿼티 파일 없음 — run_all.py 먼저 실행 필요")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "="*50)
print("  ✅ 그림 생성 완료!")
print("="*50)
print(f"""
저장 위치:
  {CWT_DIR}/Fig1_cwt_before_after_AAPL.png   ← 논문 3장
  {CWT_DIR}/Fig2_ablation_threshold_AAPL.png ← 논문 5장
  {TABLE_DIR}/Fig3_equity_curves_AAPL.png    ← 논문 4장

이 3개 파일을 Claude에 업로드하면
Word 논문에 바로 삽입해드립니다!
""")
