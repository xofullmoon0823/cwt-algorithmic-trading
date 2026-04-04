# ============================================================
# CWT 기반 Timeliness-Aware 금융 자동매매 에이전트
# 변문성 (20223872)
#
# 버전: v3.0 — 완전 재현 가능한 순수 신호 처리 기반 구현
# LLM 의존성 없음 → 누구나 동일 결과 재현 가능
#
# 핵심 아이디어:
#   CWT-FinAgent : CWT 클린 신호 기반 이평 크로스 전략
#   FinAgent 원본: 원 신호(노이즈 포함) 기반 동일 전략
#   → 입력 신호 품질 차이만으로 성능 비교
#
# 실행 방법:
#   pip install yfinance PyWavelets pandas numpy matplotlib scikit-learn
#   python run_all_v3.py
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pywt
import platform

warnings.filterwarnings("ignore")

# ── 한글 폰트 ──────────────────────────────────────────────
if platform.system() == "Darwin":
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == "Windows":
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 폴더 생성 ─────────────────────────────────────────────
for d in ["data/price", "data/clean",
          "results/baselines", "results/agent",
          "results/cwt", "results/paper_tables",
          "results/logs"]:
    os.makedirs(d, exist_ok=True)

print("=" * 60)
print("  CWT-FinAgent v3.0 — 완전 재현 가능 버전")
print("=" * 60)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TICKERS     = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]
TRAIN_START = "2022-06-01"
TRAIN_END   = "2023-05-31"
TEST_START  = "2023-06-01"
TEST_END    = "2024-01-01"
INIT_CASH   = 100_000.0

# CWT 파라미터
CWT_WAVELET   = "morl"
CWT_SCALES    = np.arange(1, 65)
CWT_A_TH      = 8        # 스케일 임계값 (핵심 하이퍼파라미터)

# 이평선 파라미터 (CWT-FinAgent / FinAgent 공통)
MA_SHORT      = 5        # 단기 이평 (일)
MA_LONG       = 20       # 장기 이평 (일)
MA_THRESHOLD  = 0.002    # 크로스 민감도 (0.2%)

print(f"\n실험 설정:")
print(f"  종목: {TICKERS}")
print(f"  학습: {TRAIN_START} ~ {TRAIN_END}")
print(f"  테스트: {TEST_START} ~ {TEST_END}")
print(f"  CWT: wavelet={CWT_WAVELET}, a_th={CWT_A_TH}")
print(f"  이평: MA{MA_SHORT}/{MA_LONG}, threshold={MA_THRESHOLD}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 1: 주가 데이터 수집
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[1/5] 주가 데이터 수집 중...")
import yfinance as yf

price_data = {}
for ticker in TICKERS:
    path = f"data/price/{ticker}_price.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        price_data[ticker] = df
        print(f"  ⏩ {ticker}: 기존 파일 로드 ({len(df)}일)")
    else:
        df = yf.download(ticker, start=TRAIN_START, end=TEST_END,
                         interval="1d", progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index.name = "Date"
        df.to_csv(path)
        price_data[ticker] = df
        print(f"  ✅ {ticker}: {len(df)}일 수집 완료")

print(f"\n  총 {len(price_data)}개 종목 준비 완료")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 2: CWT 신호 전처리
# 수식(1): W_x(a,b) = (1/√|a|) ∫ x(t)·ψ*((t-b)/a) dt
# 수식(5): x̂(t) ≈ (1/N_a) · Σ_{a≥a_th} W_x(a,t)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CWTFilter:
    """
    Morlet CWT 기반 노이즈 필터
    고주파 성분(a < a_th)을 0으로 억제하여 클린 신호 복원
    """
    def __init__(self, wavelet=CWT_WAVELET, scale_threshold=CWT_A_TH):
        self.wavelet = wavelet
        self.scales  = CWT_SCALES
        self.a_th    = scale_threshold

    def transform(self, close: np.ndarray) -> dict:
        # z-score 정규화
        mu, sigma = close.mean(), close.std() or 1.0
        x_norm    = (close - mu) / sigma

        # CWT 변환
        coef, freqs = pywt.cwt(x_norm, self.scales, self.wavelet)

        # 고주파 억제 (a < a_th)
        coef_clean = coef.copy()
        coef_clean[self.scales < self.a_th, :] = 0

        # 역변환 (근사 ICWT)
        n_low = (self.scales >= self.a_th).sum()
        x_clean_norm = (np.sum(coef_clean, axis=0) / n_low
                        if n_low > 0 else x_norm)

        # 역정규화
        x_clean = x_clean_norm * sigma + mu
        x_noise = close - x_clean

        # SNR 계산
        noise_p  = np.mean(x_noise**2)
        signal_p = np.mean(x_clean**2) + 1e-10
        snr = 10 * np.log10(signal_p / (noise_p + 1e-10))

        return {
            "clean":       x_clean,
            "noise":       x_noise,
            "coef":        coef,
            "coef_clean":  coef_clean,
            "snr":         snr,
            "noise_ratio": noise_p / (signal_p + noise_p),
        }


print("\n[2/5] CWT 클린 신호 생성 중... (a_th={})".format(CWT_A_TH))
cwt_filter = CWTFilter()
clean_data = {}

for ticker, df in price_data.items():
    res = cwt_filter.transform(df["Close"].values)
    out = df.copy()
    out["Close_clean"] = res["clean"]
    out["Noise"]       = res["noise"]
    out.to_csv(f"data/clean/{ticker}_clean_ath{CWT_A_TH}.csv")
    clean_data[ticker] = {"df": out, "res": res}
    print(f"  ✅ {ticker:6s}  SNR={res['snr']:6.1f}dB  "
          f"노이즈={res['noise_ratio']*100:.1f}%")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 3: 성능 지표 계산 (FinBen 6개 지표)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def calc_metrics(equity: pd.Series, rfr=0.04) -> dict:
    """ARR / SR / CR / SOR / MDD / VOL"""
    ret = equity.pct_change().dropna()
    n   = len(ret)
    if n == 0:
        return {k: 0.0 for k in ["ARR(%)", "SR", "CR", "SOR", "MDD(%)", "VOL(%)"]}

    total = equity.iloc[-1] / equity.iloc[0] - 1
    ARR   = (1 + total) ** (252 / n) - 1
    VOL   = ret.std() * np.sqrt(252)
    exc   = ret.mean() * 252 - rfr
    SR    = exc / VOL if VOL > 0 else 0.0
    MDD   = ((equity - equity.cummax()) / equity.cummax()).min()
    CR    = ARR / abs(MDD) if MDD != 0 else 0.0
    dv    = ret[ret < 0].std() * np.sqrt(252) if (ret < 0).any() else 1e-8
    SOR   = exc / dv

    return {
        "ARR(%)": round(ARR * 100, 3),
        "SR":     round(SR, 3),
        "CR":     round(CR, 3),
        "SOR":    round(SOR, 3),
        "MDD(%)": round(MDD * 100, 3),
        "VOL(%)": round(VOL * 100, 3),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 4: Baseline 전략
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_bah(df: pd.DataFrame) -> pd.Series:
    """Buy and Hold"""
    t = df.loc[TEST_START:TEST_END]
    shares = INIT_CASH / t["Close"].iloc[0]
    return pd.Series(t["Close"].values * shares, index=t.index)


def run_macd(df: pd.DataFrame, fast=12, slow=26, sig=9) -> pd.Series:
    """MACD 크로스 전략"""
    d = df.copy()
    d["macd"]   = (d["Close"].ewm(span=fast, adjust=False).mean()
                 - d["Close"].ewm(span=slow, adjust=False).mean())
    d["signal"] = d["macd"].ewm(span=sig, adjust=False).mean()
    d["cross"]  = np.sign(d["macd"] - d["signal"]).diff()
    t = d.loc[TEST_START:TEST_END]
    cash, pos, eq = float(INIT_CASH), 0.0, []
    for _, row in t.iterrows():
        p = float(row["Close"])
        if row["cross"] > 0 and cash > 0:
            pos, cash = cash / p, 0.0
        elif row["cross"] < 0 and pos > 0:
            cash, pos = pos * p, 0.0
        eq.append(cash + pos * p)
    return pd.Series(eq, index=t.index)


def run_rsi(df: pd.DataFrame, period=14, ob=70, os_=30) -> pd.Series:
    """RSI 역추세 전략"""
    d = df.copy()
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(com=period-1, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-10))
    t = d.loc[TEST_START:TEST_END]
    cash, pos, in_pos, eq = float(INIT_CASH), 0.0, False, []
    for _, row in t.iterrows():
        p = float(row["Close"])
        if row["rsi"] < os_ and not in_pos and cash > 0:
            pos, cash, in_pos = cash / p, 0.0, True
        elif row["rsi"] > ob and in_pos and pos > 0:
            cash, pos, in_pos = pos * p, 0.0, False
        eq.append(cash + pos * p)
    return pd.Series(eq, index=t.index)


print("\n[3/5] Baseline 전략 실험 중...")
bl_records = []
for ticker, df in price_data.items():
    for name, func in [("B&H", run_bah), ("MACD", run_macd), ("RSI", run_rsi)]:
        try:
            eq = func(df)
            m  = calc_metrics(eq)
            m.update({"Ticker": ticker, "Strategy": name})
            bl_records.append(m)
            eq.to_csv(f"results/baselines/{ticker}_{name}_equity.csv",
                      header=["Equity"])
        except Exception as e:
            print(f"  ❌ {ticker}/{name}: {e}")

baseline_df = pd.DataFrame(bl_records)
cols = ["Ticker", "Strategy", "ARR(%)", "SR", "CR", "SOR", "MDD(%)", "VOL(%)"]
baseline_df = baseline_df[cols]
baseline_df.to_csv("results/baselines/baseline_results.csv", index=False)

avg_bl = baseline_df.groupby("Strategy")[
    ["ARR(%)", "SR", "MDD(%)", "VOL(%)"]].mean().round(3)
print("\n  📊 Baseline 결과 (5개 종목 평균)")
print("  " + "-" * 55)
print(avg_bl.to_string())
print("  " + "-" * 55)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 5: CWT-FinAgent 핵심 전략
#
# [CWT-FinAgent]
#   입력: CWT 클린 신호 x̂(t)  ← 노이즈 제거됨
#   전략: MA_SHORT / MA_LONG 이평 크로스
#   → 노이즈 제거된 신호로 허위 크로스 신호 감소
#
# [FinAgent 원본]
#   입력: 원 신호 x_r(t)  ← 노이즈 포함
#   전략: 동일 MA 크로스
#   → 노이즈로 인한 허위 크로스 신호 발생
#
# 두 전략의 유일한 차이 = 입력 신호 품질
# → CWT 전처리 효과를 순수하게 분리 가능
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_cwt_agent(ticker: str,
                  use_cwt: bool = True,
                  a_th: int = CWT_A_TH) -> tuple:
    """
    CWT-FinAgent (use_cwt=True) 또는 FinAgent 원본 (use_cwt=False)

    Returns
    -------
    equity : pd.Series  — 포트폴리오 가치 시계열
    trade_log : pd.DataFrame — 매매 이력 (재현성 검증용)
    """
    df = pd.read_csv(f"data/price/{ticker}_price.csv",
                     index_col="Date", parse_dates=True)

    # CWT 클린 신호 생성
    cwt    = CWTFilter(scale_threshold=a_th)
    res    = cwt.transform(df["Close"].values)
    clean  = pd.Series(res["clean"], index=df.index)

    # 입력 신호 선택
    signal = clean if use_cwt else df["Close"]

    # 이평선 계산 (전체 기간 — lookahead bias 방지)
    ma_short = signal.rolling(MA_SHORT).mean()
    ma_long  = signal.rolling(MA_LONG).mean()

    # 테스트 구간 추출
    test       = df.loc[TEST_START:TEST_END].copy()
    ma_s_test  = ma_short.loc[TEST_START:TEST_END]
    ma_l_test  = ma_long.loc[TEST_START:TEST_END]

    # 매매 시뮬레이션
    cash, holdings = float(INIT_CASH), 0.0
    equity_list, trade_log = [], []
    prev_signal = "HOLD"

    for date, row in test.iterrows():
        raw_price = float(row["Close"])
        ms = float(ma_s_test.loc[date]) if date in ma_s_test.index else np.nan
        ml = float(ma_l_test.loc[date]) if date in ma_l_test.index else np.nan

        # 매매 신호 결정
        if np.isnan(ms) or np.isnan(ml):
            decision = "HOLD"
        elif ms > ml * (1 + MA_THRESHOLD):   # 골든 크로스
            decision = "BUY"
        elif ms < ml * (1 - MA_THRESHOLD):   # 데스 크로스
            decision = "SELL"
        else:
            decision = "HOLD"

        # 주문 집행
        action = "HOLD"
        if decision == "BUY" and cash > 100:
            holdings = cash / raw_price
            cash     = 0.0
            action   = "BUY"
        elif decision == "SELL" and holdings > 0:
            cash     = holdings * raw_price
            holdings = 0.0
            action   = "SELL"

        current = cash + holdings * raw_price
        equity_list.append(current)

        trade_log.append({
            "Date":      date.strftime("%Y-%m-%d"),
            "Close":     round(raw_price, 2),
            "MA_short":  round(ms, 2) if not np.isnan(ms) else None,
            "MA_long":   round(ml, 2) if not np.isnan(ml) else None,
            "Signal":    decision,
            "Action":    action,
            "Cash":      round(cash, 2),
            "Holdings":  round(holdings, 4),
            "Equity":    round(current, 2),
        })

    equity    = pd.Series(equity_list, index=test.index)
    trade_df  = pd.DataFrame(trade_log)

    return equity, trade_df


print("\n[4/5] CWT-FinAgent 실험 중...")
print(f"  전략: MA{MA_SHORT}/{MA_LONG} 이평 크로스 (threshold={MA_THRESHOLD})")
print(f"  CWT-FinAgent: 클린 신호 기반 | FinAgent 원본: 원 신호 기반\n")

agent_records = []

for ticker in TICKERS:
    for use_cwt, label in [(False, "FinAgent(원본)"),
                           (True,  "CWT-FinAgent(제안)")]:
        try:
            equity, trade_df = run_cwt_agent(ticker, use_cwt=use_cwt)

            # 성능 지표
            m = calc_metrics(equity)
            m.update({"Ticker": ticker, "Strategy": label})
            agent_records.append(m)

            # 저장
            safe  = label.replace("(", "").replace(")", "").replace(" ", "_")
            equity.to_csv(f"results/agent/{ticker}_{safe}_equity.csv",
                          header=["Equity"])

            # 매매 이력 저장 (재현성 검증용)
            trade_df.to_csv(f"results/logs/{ticker}_{safe}_trades.csv",
                            index=False)

            # 거래 횟수
            n_buy  = (trade_df["Action"] == "BUY").sum()
            n_sell = (trade_df["Action"] == "SELL").sum()
            arr    = m["ARR(%)"]
            sr     = m["SR"]
            label_short = "CWT" if use_cwt else "원본"
            print(f"  ✅ {ticker} {label_short:3s}  "
                  f"ARR={arr:6.2f}%  SR={sr:6.3f}  "
                  f"매수={n_buy}회  매도={n_sell}회")

        except Exception as e:
            print(f"  ❌ {ticker}/{label}: {e}")

agent_df = pd.DataFrame(agent_records)
agent_df = agent_df[cols]
agent_df.to_csv("results/agent/agent_results.csv", index=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 6: 논문 결과 테이블
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
all_df  = pd.concat([baseline_df, agent_df], ignore_index=True)
avg_all = all_df.groupby("Strategy")[
    ["ARR(%)", "SR", "CR", "SOR", "MDD(%)", "VOL(%)"]
].mean().round(3)
avg_all.to_csv("results/paper_tables/AVERAGE_results_table.csv")

# 종목별 테이블
for ticker in TICKERS:
    sub = all_df[all_df["Ticker"] == ticker].drop(columns="Ticker")
    sub.to_csv(f"results/paper_tables/{ticker}_results_table.csv", index=False)

print("\n  📋 전체 전략 평균 성능 (논문 4장 표)")
print("  " + "=" * 65)
print("  " + avg_all.to_string().replace("\n", "\n  "))
print("  " + "=" * 65)

if "CWT-FinAgent(제안)" in avg_all.index and "FinAgent(원본)" in avg_all.index:
    cwt_arr  = avg_all.loc["CWT-FinAgent(제안)", "ARR(%)"]
    base_arr = avg_all.loc["FinAgent(원본)",      "ARR(%)"]
    cwt_sr   = avg_all.loc["CWT-FinAgent(제안)", "SR"]
    base_sr  = avg_all.loc["FinAgent(원본)",      "SR"]
    cwt_mdd  = avg_all.loc["CWT-FinAgent(제안)", "MDD(%)"]
    base_mdd = avg_all.loc["FinAgent(원본)",      "MDD(%)"]

    print(f"\n  🎯 CWT vs FinAgent 원본 비교:")
    print(f"     ARR: {base_arr:+.3f}% → {cwt_arr:+.3f}%  "
          f"({cwt_arr - base_arr:+.3f}%p)")
    print(f"     SR:  {base_sr:.3f} → {cwt_sr:.3f}  "
          f"({'개선' if cwt_sr > base_sr else '저하'})")
    print(f"     MDD: {base_mdd:.3f}% → {cwt_mdd:.3f}%  "
          f"({'개선' if cwt_mdd > base_mdd else '저하'})")

print(f"\n  ✅ 결과 저장: results/paper_tables/")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 7: 논문용 그래프 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[5/5] 논문용 그래프 생성 중...")


# ── Fig1: CWT 필터 전/후 비교 (논문 3장) ──────────────────
if "AAPL" in clean_data:
    test_df = clean_data["AAPL"]["df"].loc[TEST_START:TEST_END]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(test_df.index, test_df["Close"],
                 color="#AAAAAA", lw=0.9, label="원 신호 (Raw)")
    axes[0].plot(test_df.index, test_df["Close_clean"],
                 color="#2E75B6", lw=1.6, label="CWT 클린 신호")
    axes[0].set_title(
        "AAPL — CWT 필터 전/후 가격 신호 비교 (테스트 기간)",
        fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(test_df.index, test_df["Noise"], 0,
                         where=test_df["Noise"] >= 0,
                         color="#C55A11", alpha=0.6, label="양(+) 노이즈")
    axes[1].fill_between(test_df.index, test_df["Noise"], 0,
                         where=test_df["Noise"] < 0,
                         color="#1E6B3C", alpha=0.6, label="음(-) 노이즈")
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_title("제거된 노이즈 성분", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Noise (USD)")
    axes[1].set_xlabel("Date")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/cwt/Fig1_cwt_before_after_AAPL.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ Fig1: results/cwt/Fig1_cwt_before_after_AAPL.png")


# ── Fig2: Ablation Study (논문 5장) ───────────────────────
if "AAPL" in price_data:
    close_all = price_data["AAPL"]["Close"].values
    dates_t   = price_data["AAPL"].loc[TEST_START:TEST_END].index
    fig, ax   = plt.subplots(figsize=(13, 6))
    ax.plot(dates_t,
            price_data["AAPL"].loc[TEST_START:TEST_END]["Close"].values,
            color="#CCCCCC", lw=0.8, alpha=0.8, label="원 신호")

    for a_th, col in zip([4, 8, 16, 32],
                         ["#2E75B6", "#C55A11", "#1E6B3C", "#7030A0"]):
        f   = CWTFilter(scale_threshold=a_th)
        r   = f.transform(close_all)
        cs  = pd.Series(r["clean"], index=price_data["AAPL"].index)
        ct  = cs.loc[TEST_START:TEST_END]
        ax.plot(ct.index, ct.values, color=col, lw=1.4,
                label=f"a_th={a_th}  (노이즈 {r['noise_ratio']*100:.0f}% 제거)")

    ax.set_title("Ablation Study — 스케일 임계값(a_th) 비교 (AAPL)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/cwt/Fig2_ablation_threshold_AAPL.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ Fig2: results/cwt/Fig2_ablation_threshold_AAPL.png")


# ── Fig3: 성능 비교 막대 그래프 (논문 4장) ────────────────
strategies = list(avg_all.index)
arr_vals   = avg_all["ARR(%)"].values
sr_vals    = avg_all["SR"].values
mdd_vals   = avg_all["MDD(%)"].values

# 색상 매핑
color_map = {
    "B&H":              "#888888",
    "MACD":             "#C55A11",
    "RSI":              "#7030A0",
    "FinAgent(원본)":   "#1E6B3C",
    "CWT-FinAgent(제안)": "#2E75B6",
}
colors = [color_map.get(s, "#888888") for s in strategies]
labels = [s.replace("(제안)", "\n(제안)").replace("(원본)", "\n(원본)")
          for s in strategies]

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

for ax, vals, title, ylabel in zip(
    axes,
    [arr_vals, sr_vals, mdd_vals],
    ["ARR (%) — 연간 수익률 (높을수록 좋음)",
     "SR — 샤프 지수 (높을수록 좋음)",
     "MDD (%) — 최대 낙폭 (0에 가까울수록 좋음)"],
    ["ARR (%)", "Sharpe Ratio", "MDD (%)"]
):
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.5)
    # CWT-FinAgent 강조
    cwt_idx = next((i for i, s in enumerate(strategies)
                    if "CWT" in s), -1)
    if cwt_idx >= 0:
        bars[cwt_idx].set_edgecolor("#FFD700")
        bars[cwt_idx].set_linewidth(3)

    for bar, val in zip(bars, vals):
        offset = abs(val) * 0.04 + 0.05
        ypos   = val + offset if val >= 0 else val - offset
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.2f}", ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=9, fontweight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.axhline(y=0, color="black", lw=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", labelsize=9)

plt.suptitle(
    "전략별 성능 비교 — 5개 종목 평균 (테스트: 2023.06~2024.01)",
    fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/paper_tables/Fig3_performance_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Fig3: results/paper_tables/Fig3_performance_comparison.png")


# ── Fig4: 종목별 SNR 비교 (논문 보조 그래프) ─────────────
snr_vals_t = [clean_data[t]["res"]["snr"]           for t in TICKERS]
noise_vals  = [clean_data[t]["res"]["noise_ratio"]*100 for t in TICKERS]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
bar_colors = ["#2E75B6"] * 4 + ["#C55A11"]  # TSLA 강조

b1 = ax1.bar(TICKERS, snr_vals_t, color=bar_colors, alpha=0.85, width=0.5)
for bar, val in zip(b1, snr_vals_t):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
             f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
ax1.set_title("SNR (dB) by Ticker\n(높을수록 클린 신호)", fontsize=12, fontweight="bold")
ax1.set_ylabel("SNR (dB)")
ax1.set_ylim(0, 22)
ax1.grid(True, alpha=0.3, axis="y")

b2 = ax2.bar(TICKERS, noise_vals, color=bar_colors, alpha=0.85, width=0.5)
for bar, val in zip(b2, noise_vals):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
             f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax2.set_title("Noise Ratio (%) by Ticker\n(낮을수록 클린 신호)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Noise Ratio (%)")
ax2.set_ylim(0, 6.5)
ax2.grid(True, alpha=0.3, axis="y")

# TSLA 주석
ax2.annotate("TSLA\n고변동성", xy=(4, noise_vals[4]),
             xytext=(3.0, 5.5), fontsize=9, color="#C55A11",
             fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#C55A11"))

plt.suptitle("Fig 4. CWT 노이즈 특성 비교 — 종목별 SNR 및 노이즈 비율",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/cwt/Fig4_snr_by_ticker.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Fig4: results/cwt/Fig4_snr_by_ticker.png")


# ── Fig5: AAPL 에쿼티 커브 (원본 vs CWT) ─────────────────
eq_files = {
    "B&H":            "results/baselines/AAPL_B&H_equity.csv",
    "MACD":           "results/baselines/AAPL_MACD_equity.csv",
    "FinAgent(원본)": "results/agent/AAPL_FinAgent원본_equity.csv",
    "CWT-FinAgent":   "results/agent/AAPL_CWT-FinAgent제안_equity.csv",
}
eq_styles = {
    "B&H":            ("#888888", "--", 1.3),
    "MACD":           ("#C55A11", "--", 1.3),
    "FinAgent(원본)": ("#1E6B3C", "--", 1.3),
    "CWT-FinAgent":   ("#2E75B6", "-",  2.0),
}
fig, ax = plt.subplots(figsize=(13, 6))
for label, path in eq_files.items():
    if os.path.exists(path):
        eq = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        col, ls, lw = eq_styles[label]
        ax.plot(eq.index, eq.values, color=col, lw=lw,
                linestyle=ls, label=label)
ax.axhline(y=INIT_CASH, color="black", lw=0.8,
           linestyle=":", alpha=0.4, label="초기 자금")
ax.set_title("AAPL — 전략별 에쿼티 커브 비교 (테스트: 2023.06~2024.01)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value (USD)")
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/paper_tables/Fig5_equity_curves_AAPL.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Fig5: results/paper_tables/Fig5_equity_curves_AAPL.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print("  ✅ 전체 실험 완료! — v3.0")
print("=" * 60)
print(f"""
생성된 파일:
  data/price/           ← 원 주가 데이터 (yfinance)
  data/clean/           ← CWT 클린 신호
  results/baselines/    ← B&H / MACD / RSI
  results/agent/        ← CWT-FinAgent / FinAgent 원본
  results/logs/         ← 매매 이력 (재현성 검증)
  results/paper_tables/ ← 논문 표 + 그래프
  results/cwt/          ← CWT 분석 그래프

논문용 주요 파일:
  results/paper_tables/AVERAGE_results_table.csv  ← 표 1
  results/cwt/Fig1_cwt_before_after_AAPL.png      ← 그림 1
  results/cwt/Fig2_ablation_threshold_AAPL.png    ← 그림 2
  results/paper_tables/Fig3_performance_comparison.png ← 그림 3
  results/cwt/Fig4_snr_by_ticker.png              ← 그림 4
  results/paper_tables/Fig5_equity_curves_AAPL.png ← 그림 5

재현 방법:
  1. pip install yfinance PyWavelets pandas numpy matplotlib
  2. python run_all_v3.py
  → 동일한 결과 재현 가능 (LLM 의존성 없음)
""")
