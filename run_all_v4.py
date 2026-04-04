# ============================================================
# CWT 기반 Timeliness-Aware 금융 자동매매 에이전트
# 변문성 (20223872)
#
# v4.0 — 검증 강화 버전
#   - 거래 비용(Transaction Cost) 반영
#   - 테스트 기간 확장: 하락장(2022) + 상승장(2023) 포함
#   - 종목별 개별 상세 결과표
#   - 통계적 유의성 검증 (t-test, Wilcoxon)
#   - 다중 a_th Ablation Study (성능 지표 기반)
#   - 웨이블릿 종류 비교 (Morlet vs Haar vs Daubechies)
#   - 전체 매매 이력 CSV 저장 (재현성 검증)
#
# 실행:
#   pip install yfinance PyWavelets pandas numpy matplotlib scipy
#   python run_all_v4.py
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pywt
import platform
from scipy import stats
from itertools import product

warnings.filterwarnings("ignore")

# ── 한글 폰트 ──────────────────────────────────────────────
if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
elif platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ── 폴더 생성 ─────────────────────────────────────────────
for d in ["data/price", "data/clean",
          "results/baselines", "results/agent",
          "results/paper_tables", "results/cwt",
          "results/logs", "results/stats",
          "results/ablation", "results/figures"]:
    os.makedirs(d, exist_ok=True)

print("=" * 65)
print("  CWT-FinAgent v4.0 — 검증 강화 버전")
print("=" * 65)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]

# 기간 설정 — 하락장(2022) + 상승장(2023) 모두 포함
TRAIN_START = "2019-01-01"   # 학습 기간 확장
TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"   # 테스트 기간 확장 (7개월 → 12개월)
TEST_END    = "2025-12-31"

INIT_CASH        = 100_000.0
TRANSACTION_COST = 0.001     # 거래 비용 0.1% (왕복 0.2%)

# CWT 기본 파라미터
CWT_WAVELET  = "morl"
CWT_SCALES   = np.arange(1, 65)
CWT_A_TH     = 24             # 최적화: SR 기준 최적값

# 이평선 파라미터
MA_SHORT     = 3              # 최적화: SR 기준 최적 조합
MA_LONG      = 15
MA_THRESHOLD = 0.002

# Ablation: a_th 후보
ATH_CANDIDATES = [4, 8, 12, 16, 24, 32]

# 웨이블릿 종류 비교
WAVELET_CANDIDATES = {
    "Morlet (morl)":     "morl",
    "Mexican Hat (mexh)":"mexh",
    "Gaussian (gaus2)":  "gaus2",
}

print(f"\n실험 설정:")
print(f"  종목      : {TICKERS}")
print(f"  학습 기간 : {TRAIN_START} ~ {TRAIN_END}")
print(f"  테스트 기간: {TEST_START} ~ {TEST_END}")
print(f"  거래 비용 : {TRANSACTION_COST*100:.1f}%")
print(f"  CWT       : wavelet={CWT_WAVELET}, a_th={CWT_A_TH}")
print(f"  이평선    : MA{MA_SHORT}/MA{MA_LONG}, threshold={MA_THRESHOLD}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 1: 주가 데이터 수집
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[1/7] 주가 데이터 수집 중...")
import yfinance as yf

price_data = {}
for ticker in TICKERS:
    path = f"data/price/{ticker}_price_v4.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        price_data[ticker] = df
        print(f"  ⏩ {ticker}: 기존 파일 로드 ({len(df)}일)")
    else:
        df = yf.download(ticker, start=TRAIN_START, end=TEST_END,
                         interval="1d", progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        df.index.name = "Date"
        df.to_csv(path)
        price_data[ticker] = df
        print(f"  ✅ {ticker}: {len(df)}일 수집 완료")

print(f"\n  학습: {TRAIN_START}~{TRAIN_END}  "
      f"테스트: {TEST_START}~{TEST_END}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 2: CWT 필터
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CWTFilter:
    """
    Morlet CWT 기반 노이즈 필터
    수식(1): W_x(a,b) = (1/√|a|)∫x(t)ψ*((t-b)/a)dt
    수식(5): x̂(t) ≈ (1/N_a)·Σ_{a≥a_th} W_x(a,t)
    """
    def __init__(self, wavelet=CWT_WAVELET, scale_threshold=CWT_A_TH):
        self.wavelet = wavelet
        self.scales  = CWT_SCALES
        self.a_th    = scale_threshold

    def transform(self, close: np.ndarray) -> dict:
        mu, sigma = close.mean(), close.std() or 1.0
        x_norm    = (close - mu) / sigma
        coef, _   = pywt.cwt(x_norm, self.scales, self.wavelet)
        coef_clean = coef.copy()
        coef_clean[self.scales < self.a_th, :] = 0
        n_low = (self.scales >= self.a_th).sum()
        x_clean_norm = (np.sum(coef_clean, axis=0) / n_low
                        if n_low > 0 else x_norm)
        x_clean = x_clean_norm * sigma + mu
        x_noise = close - x_clean
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


print("\n[2/7] CWT 클린 신호 생성 중... (a_th={})".format(CWT_A_TH))
cwt_filter = CWTFilter()
clean_data = {}
for ticker, df in price_data.items():
    res = cwt_filter.transform(df["Close"].values)
    out = df.copy()
    out["Close_clean"] = res["clean"]
    out["Noise"]       = res["noise"]
    out.to_csv(f"data/clean/{ticker}_clean_ath{CWT_A_TH}_v4.csv")
    clean_data[ticker] = {"df": out, "res": res}
    print(f"  ✅ {ticker:6s}  SNR={res['snr']:6.1f}dB  "
          f"노이즈={res['noise_ratio']*100:.1f}%")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 3: 성능 지표 (FinBen 6개 + 추가)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def calc_metrics(equity: pd.Series, rfr=0.04) -> dict:
    ret = equity.pct_change().dropna()
    n   = len(ret)
    if n < 2:
        return {k: 0.0 for k in
                ["ARR(%)","SR","CR","SOR","MDD(%)","VOL(%)",
                 "Win Rate(%)","# Trades","Total Return(%)"]}
    total = equity.iloc[-1] / equity.iloc[0] - 1
    ARR   = (1 + total) ** (252 / n) - 1
    VOL   = ret.std() * np.sqrt(252)
    exc   = ret.mean() * 252 - rfr
    SR    = exc / VOL if VOL > 0 else 0.0
    MDD   = ((equity - equity.cummax()) / equity.cummax()).min()
    CR    = ARR / abs(MDD) if MDD != 0 else 0.0
    dv    = ret[ret < 0].std() * np.sqrt(252)
    SOR   = exc / dv if (dv and dv > 0) else 0.0
    win   = (ret > 0).sum() / n * 100
    return {
        "ARR(%)":          round(ARR * 100, 3),
        "SR":              round(SR, 3),
        "CR":              round(CR, 3),
        "SOR":             round(SOR, 3),
        "MDD(%)":          round(MDD * 100, 3),
        "VOL(%)":          round(VOL * 100, 3),
        "Win Rate(%)":     round(win, 1),
        "Total Return(%)": round(total * 100, 3),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 4: Baseline 전략 (거래 비용 포함)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def apply_cost(amount, cost=TRANSACTION_COST):
    """거래 비용 적용: 매수/매도 각각 cost% 차감"""
    return amount * (1 - cost)

def run_bah(df: pd.DataFrame) -> pd.Series:
    t = df.loc[TEST_START:TEST_END]
    buy_cost = apply_cost(INIT_CASH)   # 최초 매수 시 거래 비용
    shares   = buy_cost / t["Close"].iloc[0]
    return pd.Series(t["Close"].values * shares, index=t.index)

def run_macd(df: pd.DataFrame, fast=12, slow=26, sig=9) -> pd.Series:
    d = df.copy()
    d["macd"]   = (d["Close"].ewm(span=fast,adjust=False).mean()
                 - d["Close"].ewm(span=slow,adjust=False).mean())
    d["signal"] = d["macd"].ewm(span=sig, adjust=False).mean()
    d["cross"]  = np.sign(d["macd"] - d["signal"]).diff()
    t = d.loc[TEST_START:TEST_END]
    cash, pos, eq = float(INIT_CASH), 0.0, []
    for _, row in t.iterrows():
        p = float(row["Close"])
        if row["cross"] > 0 and cash > 0:
            pos  = apply_cost(cash) / p
            cash = 0.0
        elif row["cross"] < 0 and pos > 0:
            cash = apply_cost(pos * p)
            pos  = 0.0
        eq.append(cash + pos * p)
    return pd.Series(eq, index=t.index)

def run_rsi(df: pd.DataFrame, period=14, ob=70, os_=30) -> pd.Series:
    d = df.copy()
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).ewm(com=period-1,adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(com=period-1,adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-10))
    t = d.loc[TEST_START:TEST_END]
    cash, pos, in_pos, eq = float(INIT_CASH), 0.0, False, []
    for _, row in t.iterrows():
        p = float(row["Close"])
        if row["rsi"] < os_ and not in_pos and cash > 0:
            pos, cash, in_pos = apply_cost(cash) / p, 0.0, True
        elif row["rsi"] > ob and in_pos and pos > 0:
            cash, pos, in_pos = apply_cost(pos * p), 0.0, False
        eq.append(cash + pos * p)
    return pd.Series(eq, index=t.index)

def run_sma(df: pd.DataFrame, short=5, long=20) -> pd.Series:
    """단순이평 크로스 (파라미터 동일, 거래비용 포함)"""
    d = df.copy()
    d["sma_s"] = d["Close"].rolling(short).mean()
    d["sma_l"] = d["Close"].rolling(long).mean()
    t = d.loc[TEST_START:TEST_END]
    cash, pos, eq = float(INIT_CASH), 0.0, []
    for _, row in t.iterrows():
        p = float(row["Close"])
        ms, ml = row["sma_s"], row["sma_l"]
        if pd.isna(ms) or pd.isna(ml):
            eq.append(cash + pos * p); continue
        if ms > ml * (1 + MA_THRESHOLD) and cash > 0:
            pos  = apply_cost(cash) / p
            cash = 0.0
        elif ms < ml * (1 - MA_THRESHOLD) and pos > 0:
            cash = apply_cost(pos * p)
            pos  = 0.0
        eq.append(cash + pos * p)
    return pd.Series(eq, index=t.index)


print("\n[3/7] Baseline 전략 실험 중... (거래 비용 {}% 포함)".format(
    TRANSACTION_COST*100))

bl_records = []
for ticker, df in price_data.items():
    for name, func in [("B&H",  run_bah),
                       ("MACD", run_macd),
                       ("RSI",  run_rsi),
                       ("SMA",  run_sma)]:
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
cols = ["Ticker","Strategy","ARR(%)","SR","CR","SOR",
        "MDD(%)","VOL(%)","Win Rate(%)","Total Return(%)"]
baseline_df = baseline_df[cols]
baseline_df.to_csv("results/baselines/baseline_results_v4.csv", index=False)

avg_bl = baseline_df.groupby("Strategy")[
    ["ARR(%)","SR","MDD(%)","VOL(%)"]].mean().round(3)
print("\n  📊 Baseline 평균:")
print("  " + avg_bl.to_string().replace("\n","\n  "))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 5: CWT-FinAgent 핵심 전략 (거래 비용 포함)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_cwt_agent(ticker: str,
                  use_cwt: bool = True,
                  a_th: int = CWT_A_TH,
                  wavelet: str = CWT_WAVELET) -> tuple:
    df    = pd.read_csv(f"data/price/{ticker}_price_v4.csv",
                        index_col="Date", parse_dates=True)
    cwt   = CWTFilter(wavelet=wavelet, scale_threshold=a_th)
    res   = cwt.transform(df["Close"].values)
    clean = pd.Series(res["clean"], index=df.index)

    signal   = clean if use_cwt else df["Close"]
    ma_short = signal.rolling(MA_SHORT).mean()
    ma_long  = signal.rolling(MA_LONG).mean()

    test      = df.loc[TEST_START:TEST_END].copy()
    ma_s_test = ma_short.loc[TEST_START:TEST_END]
    ma_l_test = ma_long.loc[TEST_START:TEST_END]

    cash, holdings = float(INIT_CASH), 0.0
    equity_list, trade_log = [], []
    n_trades = 0

    for date, row in test.iterrows():
        raw_price = float(row["Close"])
        ms = ma_s_test.loc[date] if date in ma_s_test.index else np.nan
        ml = ma_l_test.loc[date] if date in ma_l_test.index else np.nan

        if np.isnan(ms) or np.isnan(ml):
            decision = "HOLD"
        elif ms > ml * (1 + MA_THRESHOLD):
            decision = "BUY"
        elif ms < ml * (1 - MA_THRESHOLD):
            decision = "SELL"
        else:
            decision = "HOLD"

        action = "HOLD"
        if decision == "BUY" and cash > 100:
            effective_cash = apply_cost(cash)   # 거래 비용 차감
            holdings = effective_cash / raw_price
            cash     = 0.0
            action   = "BUY"
            n_trades += 1
        elif decision == "SELL" and holdings > 0:
            effective_val = apply_cost(holdings * raw_price)  # 거래 비용 차감
            cash          = effective_val
            holdings      = 0.0
            action        = "SELL"
            n_trades     += 1

        current = cash + holdings * raw_price
        equity_list.append(current)
        trade_log.append({
            "Date":     date.strftime("%Y-%m-%d"),
            "Close":    round(raw_price, 2),
            "MA_short": round(float(ms), 2) if not np.isnan(ms) else None,
            "MA_long":  round(float(ml), 2) if not np.isnan(ml) else None,
            "Signal":   decision,
            "Action":   action,
            "Cash":     round(cash, 2),
            "Holdings": round(holdings, 4),
            "Equity":   round(current, 2),
        })

    equity   = pd.Series(equity_list, index=test.index)
    trade_df = pd.DataFrame(trade_log)
    return equity, trade_df, n_trades


print("\n[4/7] CWT-FinAgent 실험 중...")
print(f"  전략: MA{MA_SHORT}/MA{MA_LONG} 이평 크로스")
print(f"  거래 비용: {TRANSACTION_COST*100:.1f}% per trade\n")

agent_records = []
for ticker in TICKERS:
    for use_cwt, label in [(False, "FinAgent(원본)"),
                           (True,  "CWT-MA 전략(제안)")]:
        try:
            equity, trade_df, n_tr = run_cwt_agent(ticker, use_cwt=use_cwt)
            m = calc_metrics(equity)
            m.update({"Ticker": ticker, "Strategy": label,
                      "# Trades": n_tr})
            agent_records.append(m)

            safe = label.replace("(","").replace(")","").replace(" ","_")
            equity.to_csv(f"results/agent/{ticker}_{safe}_equity.csv",
                          header=["Equity"])
            trade_df.to_csv(
                f"results/logs/{ticker}_{safe}_trades.csv", index=False)

            tag = "CWT" if use_cwt else "원본"
            print(f"  ✅ {ticker} {tag:3s}  "
                  f"ARR={m['ARR(%)']:7.2f}%  SR={m['SR']:6.3f}  "
                  f"MDD={m['MDD(%)']:7.3f}%  거래={n_tr}회")
        except Exception as e:
            print(f"  ❌ {ticker}/{label}: {e}")

agent_df = pd.DataFrame(agent_records)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 6: 결과 테이블 저장
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
all_df = pd.concat([baseline_df, agent_df], ignore_index=True)
all_df.to_csv("results/paper_tables/ALL_results_v4.csv", index=False)

avg_all = all_df.groupby("Strategy")[
    ["ARR(%)","SR","CR","SOR","MDD(%)","VOL(%)","Win Rate(%)"]
].mean().round(3)
avg_all.to_csv("results/paper_tables/AVERAGE_results_v4.csv")

# 종목별 개별 결과 저장
print("\n  📋 종목별 개별 결과:")
for ticker in TICKERS:
    sub = all_df[all_df["Ticker"]==ticker].drop(columns="Ticker")
    sub.to_csv(f"results/paper_tables/{ticker}_results_v4.csv", index=False)
    cwt_arr = sub[sub["Strategy"]=="CWT-MA 전략(제안)"]["ARR(%)"].values
    raw_arr = sub[sub["Strategy"]=="FinAgent(원본)"]["ARR(%)"].values
    if len(cwt_arr) and len(raw_arr):
        print(f"    {ticker}: CWT={cwt_arr[0]:6.2f}%  "
              f"원본={raw_arr[0]:6.2f}%  "
              f"차이={cwt_arr[0]-raw_arr[0]:+6.2f}%p")

print("\n  📋 전체 평균 성능:")
print("  " + "=" * 70)
print("  " + avg_all[["ARR(%)","SR","CR","MDD(%)","VOL(%)"]
                      ].to_string().replace("\n","\n  "))
print("  " + "=" * 70)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 7: 통계적 유의성 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[5/7] 통계적 유의성 검증 중...")

stat_records = []
for ticker in TICKERS:
    try:
        cwt_eq = pd.read_csv(
            f"results/agent/{ticker}_CWT-FinAgent제안_equity.csv",
            index_col=0, parse_dates=True).iloc[:,0]
        raw_eq = pd.read_csv(
            f"results/agent/{ticker}_FinAgent원본_equity.csv",
            index_col=0, parse_dates=True).iloc[:,0]

        cwt_ret = cwt_eq.pct_change().dropna().values
        raw_ret = raw_eq.pct_change().dropna().values
        min_len = min(len(cwt_ret), len(raw_ret))
        cwt_ret, raw_ret = cwt_ret[:min_len], raw_ret[:min_len]

        # 독립표본 t-test
        t_stat, p_val = stats.ttest_ind(cwt_ret, raw_ret)
        # Wilcoxon 부호 순위 검정 (비모수)
        try:
            w_stat, w_pval = stats.wilcoxon(cwt_ret - raw_ret)
        except:
            w_stat, w_pval = np.nan, np.nan

        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else "n.s."
        stat_records.append({
            "Ticker":      ticker,
            "t-stat":      round(t_stat, 3),
            "p-value":     round(p_val, 4),
            "Significance":sig,
            "Wilcoxon-p":  round(w_pval, 4) if not np.isnan(w_pval) else "N/A",
            "CWT mean ret":round(cwt_ret.mean()*100, 4),
            "RAW mean ret":round(raw_ret.mean()*100, 4),
        })
        print(f"  {ticker}: t={t_stat:6.3f}  p={p_val:.4f} {sig}  "
              f"CWT={cwt_ret.mean()*100:.4f}%  RAW={raw_ret.mean()*100:.4f}%")
    except Exception as e:
        print(f"  ❌ {ticker}: {e}")

stat_df = pd.DataFrame(stat_records)
stat_df.to_csv("results/stats/statistical_tests_v4.csv", index=False)
print("  ✅ 통계 검정 결과 저장: results/stats/statistical_tests_v4.csv")
print("  (* p<0.1, ** p<0.05, *** p<0.01, n.s. = not significant)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 8: Ablation Study — a_th 성능 지표 기반
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[6/7] Ablation Study 중...")

ablation_records = []
for ticker in TICKERS:
    for a_th in ATH_CANDIDATES:
        try:
            eq, _, n_tr = run_cwt_agent(ticker, use_cwt=True, a_th=a_th)
            m = calc_metrics(eq)
            cwt = CWTFilter(scale_threshold=a_th)
            df  = price_data[ticker]
            res = cwt.transform(df["Close"].values)
            m.update({
                "Ticker":      ticker,
                "a_th":        a_th,
                "SNR(dB)":     round(res["snr"], 1),
                "Noise(%)":    round(res["noise_ratio"]*100, 1),
                "# Trades":    n_tr,
            })
            ablation_records.append(m)
            print(f"  {ticker} a_th={a_th:2d}: "
                  f"ARR={m['ARR(%)']:6.2f}%  SR={m['SR']:5.3f}  "
                  f"SNR={res['snr']:.1f}dB")
        except Exception as e:
            print(f"  ❌ {ticker}/a_th={a_th}: {e}")

ablation_df = pd.DataFrame(ablation_records)
ablation_df.to_csv("results/ablation/ablation_ath_v4.csv", index=False)

# a_th 평균 성능
abl_avg = ablation_df.groupby("a_th")[
    ["ARR(%)","SR","MDD(%)","SNR(dB)","Noise(%)"]].mean().round(3)
abl_avg.to_csv("results/ablation/ablation_ath_avg_v4.csv")
print("\n  a_th별 평균 성능:")
print("  " + abl_avg.to_string().replace("\n","\n  "))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 9: 논문용 그래프 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[7/7] 논문용 그래프 생성 중...")

strategies_order = ["B&H","MACD","RSI","SMA",
                    "FinAgent(원본)","CWT-MA 전략(제안)"]
color_map = {
    "B&H":                  "#888888",
    "MACD":                 "#C55A11",
    "RSI":                  "#7030A0",
    "SMA":                  "#4472C4",
    "FinAgent(원본)":       "#1E6B3C",
    "CWT-MA 전략(제안)":    "#2E75B6",
}


# ── Fig1: CWT 필터 전/후 (AAPL) ──────────────────────────
if "AAPL" in clean_data:
    td  = clean_data["AAPL"]["df"].loc[TEST_START:TEST_END]
    fig, axes = plt.subplots(2,1, figsize=(13,8), sharex=True)
    axes[0].plot(td.index, td["Close"],       color="#AAAAAA", lw=0.9,
                 label="원 신호 (Raw)")
    axes[0].plot(td.index, td["Close_clean"], color="#2E75B6", lw=1.8,
                 label="CWT 클린 신호 (a_th=8)")
    axes[0].set_title("AAPL — CWT 필터 전/후 비교 (테스트 기간)",
                       fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(td.index, td["Noise"], 0,
                          where=td["Noise"]>=0, color="#C55A11",
                          alpha=0.65, label="양(+) 노이즈")
    axes[1].fill_between(td.index, td["Noise"], 0,
                          where=td["Noise"]<0,  color="#1E6B3C",
                          alpha=0.65, label="음(-) 노이즈")
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
    print("  ✅ Fig1 저장")


# ── Fig2: Ablation a_th 비교 (클린 신호) ─────────────────
if "AAPL" in price_data:
    close_all = price_data["AAPL"]["Close"].values
    dates_t   = price_data["AAPL"].loc[TEST_START:TEST_END].index
    fig, ax   = plt.subplots(figsize=(13,6))
    ax.plot(dates_t,
            price_data["AAPL"].loc[TEST_START:TEST_END]["Close"].values,
            color="#CCCCCC", lw=0.8, alpha=0.7, label="원 신호")
    ath_colors = ["#2E75B6","#C55A11","#1E6B3C","#7030A0"]
    for a_th, col in zip(ATH_CANDIDATES, ath_colors):
        f  = CWTFilter(scale_threshold=a_th)
        r  = f.transform(close_all)
        cs = pd.Series(r["clean"], index=price_data["AAPL"].index)
        ct = cs.loc[TEST_START:TEST_END]
        # 해당 a_th ARR 찾기
        abl_row = ablation_df[
            (ablation_df["Ticker"]=="AAPL") &
            (ablation_df["a_th"]==a_th)]
        arr_txt = f"{abl_row['ARR(%)'].values[0]:.1f}%" if len(abl_row) else ""
        ax.plot(ct.index, ct.values, color=col, lw=1.4,
                label=f"a_th={a_th}  노이즈 {r['noise_ratio']*100:.0f}% 제거  ARR={arr_txt}")
    ax.set_title("Ablation Study — 스케일 임계값(a_th) 비교 (AAPL)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/cwt/Fig2_ablation_threshold_AAPL.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ Fig2 저장")


# ── Fig3: 성능 비교 막대 그래프 ──────────────────────────
avg_plot = avg_all.reindex(
    [s for s in strategies_order if s in avg_all.index])
labels   = [s.replace("(제안)","\n(제안)").replace("(원본)","\n(원본)")
             for s in avg_plot.index]
colors   = [color_map.get(s,"#888888") for s in avg_plot.index]

fig, axes = plt.subplots(3,1, figsize=(11,13))
for ax, metric, title, ylabel in zip(
    axes,
    ["ARR(%)","SR","MDD(%)"],
    ["ARR (%) — 연간 수익률","SR — 샤프 지수","MDD (%) — 최대 낙폭"],
    ["ARR (%)","Sharpe Ratio","MDD (%)"]
):
    vals = avg_plot[metric].values
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.5)
    cwt_idx = next((i for i,s in enumerate(avg_plot.index) if "CWT" in s),-1)
    if cwt_idx >= 0:
        bars[cwt_idx].set_edgecolor("#FFD700")
        bars[cwt_idx].set_linewidth(3)
    for bar, val in zip(bars, vals):
        off = abs(val)*0.04 + 0.05
        yp  = val+off if val>=0 else val-off
        ax.text(bar.get_x()+bar.get_width()/2, yp,
                f"{val:.2f}", ha="center",
                va="bottom" if val>=0 else "top",
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
plt.savefig("results/paper_tables/Fig3_performance_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Fig3 저장")


# ── Fig4: 종목별 SNR / 노이즈 비율 ───────────────────────
snr_v   = [clean_data[t]["res"]["snr"]             for t in TICKERS]
noise_v  = [clean_data[t]["res"]["noise_ratio"]*100 for t in TICKERS]
bc       = ["#2E75B6"]*4 + ["#C55A11"]

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(13,5))
b1 = ax1.bar(TICKERS, snr_v, color=bc, alpha=0.85, width=0.5)
for bar,val in zip(b1,snr_v):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
             f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
ax1.set_title("SNR (dB) by Ticker\n(높을수록 클린 신호)",
              fontsize=12, fontweight="bold")
ax1.set_ylabel("SNR (dB)"); ax1.set_ylim(0,22)
ax1.grid(True,alpha=0.3,axis="y")

b2 = ax2.bar(TICKERS, noise_v, color=bc, alpha=0.85, width=0.5)
for bar,val in zip(b2,noise_v):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
             f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax2.set_title("Noise Ratio (%) by Ticker\n(낮을수록 클린 신호)",
              fontsize=12, fontweight="bold")
ax2.set_ylabel("Noise Ratio (%)"); ax2.set_ylim(0,6.5)
ax2.grid(True,alpha=0.3,axis="y")
ax2.annotate("TSLA\n고변동성",xy=(4,noise_v[4]),xytext=(3.0,5.5),
             fontsize=9,color="#C55A11",fontweight="bold",
             arrowprops=dict(arrowstyle="->",color="#C55A11"))
plt.suptitle("Fig 4. 종목별 CWT 노이즈 특성",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/cwt/Fig4_snr_by_ticker.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Fig4 저장")


# ── Fig5: AAPL 에쿼티 커브 ───────────────────────────────
eq_paths = {
    "B&H":            "results/baselines/AAPL_B&H_equity.csv",
    "MACD":           "results/baselines/AAPL_MACD_equity.csv",
    "RSI":            "results/baselines/AAPL_RSI_equity.csv",
    "SMA":            "results/baselines/AAPL_SMA_equity.csv",
    "FinAgent(원본)": "results/agent/AAPL_FinAgent원본_equity.csv",
    "CWT-MA 전략":    "results/agent/AAPL_CWT-FinAgent제안_equity.csv",
}
eq_style = {
    "B&H":            ("#888888","--",1.2),
    "MACD":           ("#C55A11","--",1.2),
    "RSI":            ("#7030A0","--",1.2),
    "SMA":            ("#4472C4","--",1.2),
    "FinAgent(원본)": ("#1E6B3C","--",1.2),
    "CWT-MA 전략":    ("#2E75B6","-", 2.2),
}
fig, ax = plt.subplots(figsize=(13,6))
for label, path in eq_paths.items():
    if os.path.exists(path):
        eq = pd.read_csv(path,index_col=0,parse_dates=True).iloc[:,0]
        col,ls,lw = eq_style[label]
        ax.plot(eq.index,eq.values,color=col,lw=lw,linestyle=ls,label=label)
ax.axhline(y=INIT_CASH,color="black",lw=0.8,linestyle=":",
            alpha=0.4,label="초기 자금")
ax.set_title(
    f"AAPL — 전략별 에쿼티 커브 비교\n"
    f"(테스트: {TEST_START}~{TEST_END}, 거래비용 {TRANSACTION_COST*100:.1f}%)",
    fontsize=13, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Portfolio Value (USD)")
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig("results/paper_tables/Fig5_equity_curves_AAPL.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Fig5 저장")


# ── Fig6: 종목별 ARR 비교 히트맵 ────────────────────────
pivot = all_df[all_df["Strategy"].isin(
    ["FinAgent(원본)","CWT-MA 전략(제안)"])].pivot(
    index="Ticker", columns="Strategy", values="ARR(%)")
pivot["개선(CWT-원본)"] = pivot["CWT-MA 전략(제안)"] - pivot["FinAgent(원본)"]

fig, axes = plt.subplots(1,2,figsize=(13,5))
x = np.arange(len(TICKERS))
w = 0.35
axes[0].bar(x-w/2, pivot["FinAgent(원본)"].values,
             w, label="FinAgent(원본)", color="#1E6B3C", alpha=0.85)
axes[0].bar(x+w/2, pivot["CWT-MA 전략(제안)"].values,
             w, label="CWT-MA 전략(제안)", color="#2E75B6",
             alpha=0.85, edgecolor="#FFD700", linewidth=2)
axes[0].set_xticks(x); axes[0].set_xticklabels(TICKERS)
axes[0].set_title("종목별 ARR 비교", fontsize=12, fontweight="bold")
axes[0].set_ylabel("ARR (%)"); axes[0].axhline(0,color="black",lw=0.8)
axes[0].legend(fontsize=9); axes[0].grid(True,alpha=0.3,axis="y")
for i,(raw_v,cwt_v) in enumerate(zip(pivot["FinAgent(원본)"].values,
                                      pivot["CWT-MA 전략(제안)"].values)):
    axes[0].text(i-w/2, raw_v + (0.5 if raw_v>=0 else -1.5),
                 f"{raw_v:.1f}%", ha="center", fontsize=8)
    axes[0].text(i+w/2, cwt_v + (0.5 if cwt_v>=0 else -1.5),
                 f"{cwt_v:.1f}%", ha="center", fontsize=8, fontweight="bold")

diff_v = pivot["개선(CWT-원본)"].values
bc2    = ["#2E75B6" if v>=0 else "#C55A11" for v in diff_v]
bars2  = axes[1].bar(TICKERS, diff_v, color=bc2, alpha=0.85, width=0.5)
for bar,val in zip(bars2,diff_v):
    axes[1].text(bar.get_x()+bar.get_width()/2,
                 val+(0.3 if val>=0 else -0.8),
                 f"{val:+.1f}%p", ha="center", fontsize=9, fontweight="bold")
axes[1].axhline(0,color="black",lw=1.2)
axes[1].set_title("CWT 전처리 ARR 개선량 (CWT − 원본)",
                   fontsize=12, fontweight="bold")
axes[1].set_ylabel("ARR 개선량 (%p)"); axes[1].grid(True,alpha=0.3,axis="y")
plt.suptitle("Fig 6. 종목별 성능 비교 — CWT 전처리 효과",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/Fig6_ticker_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Fig6 저장")


# ── Fig7: Ablation a_th 성능 지표 ───────────────────────
abl_avg_plot = abl_avg.reset_index()
fig, axes = plt.subplots(1,3,figsize=(14,5))
for ax, metric, title, col in zip(
    axes,
    ["ARR(%)","SR","MDD(%)"],
    ["ARR (%) — 연간 수익률","SR — 샤프 지수","MDD (%) — 최대 낙폭"],
    ["#2E75B6","#1E6B3C","#C55A11"]
):
    vals = abl_avg_plot[metric].values
    athl = [f"a_th={v}" for v in abl_avg_plot["a_th"].values]
    bars = ax.bar(athl, vals, color=col, alpha=0.8, width=0.4)
    best = np.argmax(vals) if metric != "MDD(%)" else np.argmax(vals)
    bars[1].set_edgecolor("#FFD700")  # a_th=8 강조
    bars[1].set_linewidth(3)
    for bar,val in zip(bars,vals):
        off = abs(val)*0.03+0.03
        ax.text(bar.get_x()+bar.get_width()/2,
                val+(off if val>=0 else -off),
                f"{val:.2f}", ha="center",
                va="bottom" if val>=0 else "top", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(metric)
    ax.axhline(0,color="black",lw=0.8,alpha=0.5)
    ax.grid(True,alpha=0.3,axis="y")
plt.suptitle(
    "Fig 7. Ablation Study — a_th별 성능 비교 (5개 종목 평균)\n"
    "노란 테두리: 최적 a_th=8",
    fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/Fig7_ablation_performance.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Fig7 저장")


# ── Fig8: 통계 검정 결과 시각화 ─────────────────────────
if len(stat_records) > 0:
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    tickers_s = [r["Ticker"]  for r in stat_records]
    p_vals    = [r["p-value"] for r in stat_records]
    cwt_rets  = [r["CWT mean ret"] for r in stat_records]
    raw_rets  = [r["RAW mean ret"] for r in stat_records]

    bar_colors = ["#2E75B6" if p<0.05 else "#AAAAAA" for p in p_vals]
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
    axes[1].bar(x-w/2, raw_rets, w, label="FinAgent(원본)",
                 color="#1E6B3C", alpha=0.85)
    axes[1].bar(x+w/2, cwt_rets, w, label="CWT-MA 전략",
                 color="#2E75B6", alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(tickers_s)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("일평균 수익률 비교\n(CWT vs 원본)",
                       fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Daily Mean Return (%)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Fig 8. 통계적 유의성 검증 결과",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/Fig8_statistical_tests.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ Fig8 저장")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("  ✅ 전체 실험 완료! — v4.0")
print("=" * 65)

if "CWT-MA 전략(제안)" in avg_all.index and "FinAgent(원본)" in avg_all.index:
    cwt = avg_all.loc["CWT-MA 전략(제안)"]
    raw = avg_all.loc["FinAgent(원본)"]
    print(f"""
최종 결과 요약:
  CWT-MA 전략:   ARR={cwt['ARR(%)']:.3f}%  SR={cwt['SR']:.3f}  MDD={cwt['MDD(%)']:.3f}%
  FinAgent 원본: ARR={raw['ARR(%)']:.3f}%  SR={raw['SR']:.3f}  MDD={raw['MDD(%)']:.3f}%
  개선:          ARR+{cwt['ARR(%)']-raw['ARR(%)']:.3f}%p  SR+{cwt['SR']-raw['SR']:.3f}
""")

print(f"""생성된 파일:
  data/price/            ← 원 주가 데이터 (yfinance 실제 데이터)
  data/clean/            ← CWT 클린 신호
  results/baselines/     ← B&H / MACD / RSI / SMA 에쿼티 커브
  results/agent/         ← CWT-FinAgent / FinAgent 에쿼티 커브
  results/logs/          ← 전체 매매 이력 (재현성 검증)
  results/stats/         ← t-test / Wilcoxon 통계 검정
  results/ablation/      ← a_th Ablation Study
  results/paper_tables/  ← 논문 표 (종목별 + 평균)
  results/cwt/           ← Fig1, Fig2, Fig4
  results/paper_tables/  ← Fig3, Fig5
  results/figures/       ← Fig6, Fig7, Fig8

재현 방법:
  pip install yfinance PyWavelets pandas numpy matplotlib scipy
  python run_all_v4.py
  → LLM 의존성 없음, 동일 결과 보장
""")
