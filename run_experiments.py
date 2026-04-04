#!/usr/bin/env python3
# ============================================================
# 추가 실험 3종
#   실험1 : 60일 Rolling Window 슬라이딩 검증
#   실험2 : 2022년 하락장 검증
#   실험3 : TSLA 적응형 a_th 검증
#
# 실행: python3 run_experiments.py
# ============================================================

import os, warnings
import numpy as np
import pandas as pd
import pywt
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ── 출력 폴더 ─────────────────────────────────────────────
for d in ["data/price", "results/experiments"]:
    os.makedirs(d, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 공통 설정 (run_all_v4.py 와 동일)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TICKERS          = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]
TRAIN_START      = "2019-01-01"
TRAIN_END        = "2022-12-31"
TEST_START       = "2023-01-01"
TEST_END         = "2025-12-31"
INIT_CASH        = 100_000.0
TRANSACTION_COST = 0.001          # 0.1%
CWT_WAVELET      = "morl"
CWT_SCALES       = np.arange(1, 65)
CWT_A_TH         = 24
MA_SHORT         = 3
MA_LONG          = 15
MA_THRESHOLD     = 0.002


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 공통 헬퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def apply_cost(amount):
    return amount * (1 - TRANSACTION_COST)


def calc_metrics(equity: pd.Series, rfr=0.04) -> dict:
    ret = equity.pct_change().dropna()
    n   = len(ret)
    if n < 2:
        return {"ARR(%)": 0.0, "SR": 0.0, "MDD(%)": 0.0}
    total = equity.iloc[-1] / equity.iloc[0] - 1
    ARR   = (1 + total) ** (252 / n) - 1
    VOL   = ret.std() * np.sqrt(252)
    exc   = ret.mean() * 252 - rfr
    SR    = exc / VOL if VOL > 0 else 0.0
    MDD   = ((equity - equity.cummax()) / equity.cummax()).min()
    return {
        "ARR(%)": round(ARR * 100, 3),
        "SR":     round(SR,        3),
        "MDD(%)": round(MDD * 100, 3),
    }


def cwt_clean_signal(close_arr: np.ndarray, a_th: int = CWT_A_TH) -> np.ndarray:
    """CWT 노이즈 제거 → 클린 신호 반환"""
    mu, sigma = close_arr.mean(), close_arr.std() or 1.0
    x_norm    = (close_arr - mu) / sigma
    coef, _   = pywt.cwt(x_norm, CWT_SCALES, CWT_WAVELET)
    coef_c    = coef.copy()
    coef_c[CWT_SCALES < a_th, :] = 0
    n_low     = (CWT_SCALES >= a_th).sum()
    x_c_norm  = np.sum(coef_c, axis=0) / n_low if n_low > 0 else x_norm
    return x_c_norm * sigma + mu, coef


def load_price(ticker: str) -> pd.DataFrame:
    """yfinance 다운로드 or 캐시 로드"""
    path = f"data/price/{ticker}_price_v4.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
    else:
        df = yf.download(ticker, start=TRAIN_START, end="2025-12-31",
                         interval="1d", progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        df.index.name = "Date"
        df.to_csv(path)
    return df


def run_ma_strategy(
        signal: pd.Series,
        price:  pd.Series,
        test_start: str,
        test_end:   str,
) -> pd.Series:
    """
    MA 크로스오버 매매.
    signal : MA 계산에 쓸 가격 계열 (CWT 클린 or 원본)
    price  : 실제 체결 가격
    """
    ma_s = signal.rolling(MA_SHORT).mean()
    ma_l = signal.rolling(MA_LONG).mean()

    price_t = price.loc[test_start:test_end]
    ma_s_t  = ma_s.loc[test_start:test_end]
    ma_l_t  = ma_l.loc[test_start:test_end]

    cash, hold = float(INIT_CASH), 0.0
    eq = []
    for date in price_t.index:
        p  = float(price_t.loc[date])
        ms = float(ma_s_t.loc[date]) if date in ma_s_t.index else np.nan
        ml = float(ma_l_t.loc[date]) if date in ma_l_t.index else np.nan

        if not (np.isnan(ms) or np.isnan(ml)):
            if ms > ml * (1 + MA_THRESHOLD) and cash > 100:
                hold = apply_cost(cash) / p
                cash = 0.0
            elif ms < ml * (1 - MA_THRESHOLD) and hold > 0:
                cash = apply_cost(hold * p)
                hold = 0.0

        eq.append(cash + hold * p)
    return pd.Series(eq, index=price_t.index)


def run_bah(price: pd.Series, test_start: str, test_end: str) -> pd.Series:
    t      = price.loc[test_start:test_end]
    shares = apply_cost(INIT_CASH) / float(t.iloc[0])
    return pd.Series(t.values * shares, index=t.index)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 데이터 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 65)
print("  추가 실험 3종  (CWT-FinAgent  a_th=24, MA3/15)")
print("=" * 65)

print("\n[데이터 로드 중...]")
price_data = {}
for t in TICKERS:
    price_data[t] = load_price(t)
    print(f"  {t}: {len(price_data[t])}일")

# CWT 클린 신호 (전체 기간, a_th=24)
clean_series = {}
cwt_coef_all  = {}
for t, df in price_data.items():
    clean_arr, coef = cwt_clean_signal(df["Close"].values, CWT_A_TH)
    clean_series[t] = pd.Series(clean_arr, index=df.index)
    cwt_coef_all[t] = coef


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 1: 60일 Rolling Window 슬라이딩 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("  실험1: 60일 Rolling Window 슬라이딩 검증")
print(f"  테스트 기간: {TEST_START} ~ {TEST_END}")
print("=" * 65)

WINDOW = 60  # 거래일 기준

exp1_rows = []

for ticker in TICKERS:
    df    = price_data[ticker]
    price = df["Close"]
    clean = clean_series[ticker]
    raw   = price  # CWT 미적용

    # 전체 테스트 기간 equity 사전 계산
    eq_cwt = run_ma_strategy(clean, price, TEST_START, TEST_END)
    eq_raw = run_ma_strategy(raw,   price, TEST_START, TEST_END)

    ret_cwt = eq_cwt.pct_change().dropna()
    ret_raw = eq_raw.pct_change().dropna()

    # 60일 윈도우 슬라이딩
    n = min(len(ret_cwt), len(ret_raw))
    dates_cwt = ret_cwt.index[:n]
    r_c = ret_cwt.values[:n]
    r_r = ret_raw.values[:n]

    n_windows   = 0
    cwt_wins    = 0
    diff_means  = []

    for start in range(0, n - WINDOW + 1):
        w_c = r_c[start:start + WINDOW]
        w_r = r_r[start:start + WINDOW]
        mean_c = w_c.mean()
        mean_r = w_r.mean()
        diff_means.append(mean_c - mean_r)
        n_windows += 1
        if mean_c > mean_r:
            cwt_wins += 1

    if n_windows == 0:
        print(f"  {ticker}: 윈도우 부족")
        continue

    diff_arr = np.array(diff_means)
    # t-test: 각 윈도우 diff가 0보다 큰가 (단측)
    t_stat, p_two = stats.ttest_1samp(diff_arr, 0)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    sig = "***" if p_one < 0.01 else "**" if p_one < 0.05 else "*" if p_one < 0.1 else "n.s."

    exp1_rows.append({
        "Ticker":          ticker,
        "총 윈도우 수":    n_windows,
        "CWT 우위 윈도우": cwt_wins,
        "CWT 우위 비율(%)": round(cwt_wins / n_windows * 100, 1),
        "평균 수익률차(bps)": round(diff_arr.mean() * 10000, 4),
        "t-stat":          round(t_stat, 3),
        "p-value(단측)":   round(p_one,  4),
        "유의성":          sig,
    })

    print(f"  {ticker}  윈도우={n_windows}  CWT우위={cwt_wins}/{n_windows}"
          f"({cwt_wins/n_windows*100:.1f}%)  "
          f"diff={diff_arr.mean()*10000:.2f}bps  "
          f"t={t_stat:.3f}  p={p_one:.4f} {sig}")

exp1_df = pd.DataFrame(exp1_rows)
exp1_df.to_csv("results/experiments/exp1_rolling60_v4.csv", index=False)

print("\n  [실험1 결과표]")
print(exp1_df.to_string(index=False))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 2: 2022년 하락장 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("  실험2: 2022년 하락장 검증")
print("  테스트 기간: 2022-01-01 ~ 2022-12-31")
print("=" * 65)

BEAR_START = "2022-01-01"
BEAR_END   = "2022-12-31"

exp2_rows = []

for ticker in TICKERS:
    df    = price_data[ticker]
    price = df["Close"]

    # 2022 기간 데이터 존재 여부 확인
    avail = price.loc[BEAR_START:BEAR_END]
    if len(avail) < 20:
        print(f"  {ticker}: 2022 데이터 부족 ({len(avail)}일)")
        continue

    # CWT 클린 신호는 전체 기간으로 계산 (look-ahead 없이)
    # → 2022 이전 데이터만 사용해 CWT 적용
    df_train = df.loc[:BEAR_END]
    clean_arr_2022, _ = cwt_clean_signal(df_train["Close"].values, CWT_A_TH)
    clean_2022 = pd.Series(clean_arr_2022, index=df_train.index)

    price_full = df_train["Close"]

    eq_cwt  = run_ma_strategy(clean_2022, price_full, BEAR_START, BEAR_END)
    eq_raw  = run_ma_strategy(price_full,  price_full, BEAR_START, BEAR_END)
    eq_bah  = run_bah(price_full, BEAR_START, BEAR_END)

    m_cwt = calc_metrics(eq_cwt)
    m_raw = calc_metrics(eq_raw)
    m_bah = calc_metrics(eq_bah)

    for strat, m in [("CWT-FinAgent(제안)", m_cwt),
                     ("FinAgent(원본)",     m_raw),
                     ("B&H",               m_bah)]:
        row = {"Ticker": ticker, "전략": strat}
        row.update(m)
        exp2_rows.append(row)

    print(f"  {ticker}:")
    print(f"    CWT-FinAgent : ARR={m_cwt['ARR(%)']:7.2f}%  SR={m_cwt['SR']:6.3f}  MDD={m_cwt['MDD(%)']:7.3f}%")
    print(f"    FinAgent(원본): ARR={m_raw['ARR(%)']:7.2f}%  SR={m_raw['SR']:6.3f}  MDD={m_raw['MDD(%)']:7.3f}%")
    print(f"    B&H           : ARR={m_bah['ARR(%)']:7.2f}%  SR={m_bah['SR']:6.3f}  MDD={m_bah['MDD(%)']:7.3f}%")

exp2_df = pd.DataFrame(exp2_rows)
exp2_df.to_csv("results/experiments/exp2_bear2022_v4.csv", index=False)

print("\n  [실험2 결과표]")
# 피벗 테이블: 종목 × 전략
pivot_cols = ["ARR(%)", "SR", "MDD(%)"]
for col in pivot_cols:
    pv = exp2_df.pivot(index="Ticker", columns="전략", values=col)
    # 열 순서 고정
    order = ["CWT-FinAgent(제안)", "FinAgent(원본)", "B&H"]
    pv = pv[[c for c in order if c in pv.columns]]
    print(f"\n  ▶ {col}")
    print(pv.to_string())

print("\n  전체 평균:")
avg2 = exp2_df.groupby("전략")[pivot_cols].mean().round(3)
print(avg2.to_string())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 3: TSLA 적응형 a_th 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("  실험3: TSLA 적응형 a_th 검증")
print(f"  a_th(t) = 24 × (σ_base / σ_t)")
print(f"  σ_base : 학습기간({TRAIN_START}~{TRAIN_END}) 20일 std 평균")
print(f"  σ_t    : 테스트 각 시점 20일 std")
print("=" * 65)

ticker = "TSLA"
df     = price_data[ticker]
price  = df["Close"]

# σ_base: 학습기간 TSLA 20일 rolling std 평균
train_price  = price.loc[TRAIN_START:TRAIN_END]
sigma_base   = train_price.rolling(20).std().dropna().mean()
print(f"\n  TSLA σ_base = {sigma_base:.4f}  (학습기간 20일 std 평균)")

# 전체 기간 20일 rolling std
rolling_std_all = price.rolling(20).std()

# 적응형 a_th 계열 (테스트 기간)
test_price     = price.loc[TEST_START:TEST_END]
rolling_std_t  = rolling_std_all.loc[TEST_START:TEST_END]
adaptive_ath_t = (CWT_A_TH * sigma_base / rolling_std_t).clip(1, 64)

print(f"  적응형 a_th 통계:")
print(f"    평균={adaptive_ath_t.mean():.2f}  "
      f"최솟값={adaptive_ath_t.min():.2f}  "
      f"최댓값={adaptive_ath_t.max():.2f}  "
      f"std={adaptive_ath_t.std():.2f}")

# ── 적응형 CWT 클린 신호 생성 ──────────────────────────
# CWT 변환은 전체 기간에 대해 1회 수행
close_arr = price.values
mu, sigma_arr = close_arr.mean(), close_arr.std() or 1.0
x_norm        = (close_arr - mu) / sigma_arr
coef_full, _  = pywt.cwt(x_norm, CWT_SCALES, CWT_WAVELET)

# 시점별 적응형 a_th 적용
adaptive_ath_full = pd.Series(np.nan, index=price.index)
adaptive_ath_full.update(adaptive_ath_t)
# 테스트 이전 구간은 고정값 24 사용
adaptive_ath_full.fillna(CWT_A_TH, inplace=True)

clean_adaptive_arr = np.zeros(len(close_arr))
for i in range(len(close_arr)):
    a_t   = int(round(float(adaptive_ath_full.iloc[i])))
    a_t   = max(1, min(64, a_t))
    mask  = CWT_SCALES >= a_t
    n_low = mask.sum()
    if n_low > 0:
        clean_adaptive_arr[i] = np.sum(coef_full[mask, i]) / n_low * sigma_arr + mu
    else:
        clean_adaptive_arr[i] = close_arr[i]

clean_adaptive = pd.Series(clean_adaptive_arr, index=price.index)

# ── 전략 실행 ──────────────────────────────────────────
# 1. 적응형 a_th 전략
eq_adaptive = run_ma_strategy(clean_adaptive, price, TEST_START, TEST_END)

# 2. 고정 a_th=24 전략 (CWT-FinAgent 기본)
eq_fixed    = run_ma_strategy(clean_series[ticker], price, TEST_START, TEST_END)

# 3. 기준전략 (CWT 미적용)
eq_raw      = run_ma_strategy(price, price, TEST_START, TEST_END)

# ── 결과 출력 ──────────────────────────────────────────
m_adapt  = calc_metrics(eq_adaptive)
m_fixed  = calc_metrics(eq_fixed)
m_raw    = calc_metrics(eq_raw)

exp3_rows = [
    {"전략": f"적응형 a_th  (avg≈{adaptive_ath_t.mean():.1f})", **m_adapt},
    {"전략": f"고정  a_th=24 (CWT-FinAgent)",                   **m_fixed},
    {"전략": "기준전략 (CWT 미적용)",                            **m_raw},
]
exp3_df = pd.DataFrame(exp3_rows)
exp3_df.to_csv("results/experiments/exp3_adaptive_ath_TSLA_v4.csv", index=False)

print(f"\n  [실험3 결과표 — TSLA, {TEST_START}~{TEST_END}]")
print(exp3_df.to_string(index=False))

# 추가: 분기별 적응형 a_th 분포 확인
print("\n  분기별 적응형 a_th 평균:")
quarterly = adaptive_ath_t.resample("QE").mean()
for q, v in quarterly.items():
    print(f"    {q.strftime('%Y-Q%q') if hasattr(q,'strftime') else q}: a_th ≈ {v:.2f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 최종 요약
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("  실험 완료 — 저장 위치:")
print("    results/experiments/exp1_rolling60_v4.csv")
print("    results/experiments/exp2_bear2022_v4.csv")
print("    results/experiments/exp3_adaptive_ath_TSLA_v4.csv")
print("=" * 65)
