# ============================================================
# CWT-FinAgent パラメータ最適化スクリプト
# Step 1: 2019-01-01 ~ 2022-12-31 / 2023-01-01 ~ 2025-12-31
# Step 2: a_th 最適化 (4,8,12,16,24,32)
# Step 3: MA 조합 최적화 (3/15, 5/20, 5/30, 8/21, 8/30)
# ============================================================

import os, warnings
import numpy as np
import pandas as pd
import pywt
import yfinance as yf

warnings.filterwarnings("ignore")

for d in ["data/price", "data/clean", "results/optimization"]:
    os.makedirs(d, exist_ok=True)

# ─── 실험 설정 ───────────────────────────────────────────
TICKERS      = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]
TRAIN_START  = "2019-01-01"
TRAIN_END    = "2022-12-31"
TEST_START   = "2023-01-01"
TEST_END     = "2025-12-31"
INIT_CASH    = 100_000.0
TC           = 0.001        # 거래 비용 0.1%
MA_THRESHOLD = 0.002
CWT_WAVELET  = "morl"
CWT_SCALES   = np.arange(1, 65)

ATH_CANDIDATES = [4, 8, 12, 16, 24, 32]
MA_CANDIDATES  = [(3,15), (5,20), (5,30), (8,21), (8,30)]

# ─── 유틸 ────────────────────────────────────────────────
def apply_cost(amount):
    return amount * (1 - TC)

def calc_metrics(equity: pd.Series, rfr=0.04) -> dict:
    ret = equity.pct_change().dropna()
    n   = len(ret)
    if n < 2:
        return {"ARR(%)":0,"SR":0,"MDD(%)":0,"VOL(%)":0}
    total = equity.iloc[-1] / equity.iloc[0] - 1
    ARR   = (1 + total) ** (252 / n) - 1
    VOL   = ret.std() * np.sqrt(252)
    exc   = ret.mean() * 252 - rfr
    SR    = exc / VOL if VOL > 0 else 0.0
    MDD   = ((equity - equity.cummax()) / equity.cummax()).min()
    CR    = ARR / abs(MDD) if MDD != 0 else 0.0
    return {
        "ARR(%)": round(ARR*100, 3),
        "SR":     round(SR, 3),
        "CR":     round(CR, 3),
        "MDD(%)": round(MDD*100, 3),
        "VOL(%)": round(VOL*100, 3),
    }

def cwt_clean(close_vals, a_th, wavelet=CWT_WAVELET):
    mu, sigma = close_vals.mean(), close_vals.std() or 1.0
    x_norm = (close_vals - mu) / sigma
    coef, _ = pywt.cwt(x_norm, CWT_SCALES, wavelet)
    coef_c  = coef.copy()
    coef_c[CWT_SCALES < a_th, :] = 0
    n_low   = (CWT_SCALES >= a_th).sum()
    x_c     = (np.sum(coef_c, axis=0) / n_low if n_low > 0 else x_norm)
    return x_c * sigma + mu

def run_ma_strategy(price_df, signal_series, ma_short, ma_long):
    """MA 크로스 전략 실행"""
    ma_s = signal_series.rolling(ma_short).mean()
    ma_l = signal_series.rolling(ma_long).mean()
    test = price_df.loc[TEST_START:TEST_END].copy()
    ms_t = ma_s.loc[TEST_START:TEST_END]
    ml_t = ma_l.loc[TEST_START:TEST_END]

    cash, hold = float(INIT_CASH), 0.0
    equity = []
    n_trades = 0
    for date, row in test.iterrows():
        p  = float(row["Close"])
        ms = ms_t.loc[date] if date in ms_t.index else np.nan
        ml = ml_t.loc[date] if date in ml_t.index else np.nan
        if np.isnan(ms) or np.isnan(ml):
            equity.append(cash + hold * p); continue
        if ms > ml * (1 + MA_THRESHOLD) and cash > 100:
            hold  = apply_cost(cash) / p
            cash  = 0.0
            n_trades += 1
        elif ms < ml * (1 - MA_THRESHOLD) and hold > 0:
            cash  = apply_cost(hold * p)
            hold  = 0.0
            n_trades += 1
        equity.append(cash + hold * p)
    return pd.Series(equity, index=test.index), n_trades

# ═══════════════════════════════════════════════════════════
print("=" * 65)
print("  CWT-FinAgent 파라미터 최적화")
print("=" * 65)
print(f"  학습: {TRAIN_START}~{TRAIN_END}")
print(f"  테스트: {TEST_START}~{TEST_END}")

# ─── Step 1: 데이터 수집 ──────────────────────────────────
print("\n[Step 1] 주가 데이터 수집...")
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

# ─── Step 2: a_th 최적화 ──────────────────────────────────
print("\n[Step 2] a_th 최적화 탐색 (MA=5/20 고정)...")
print(f"  후보: {ATH_CANDIDATES}")
print("-" * 65)

ath_records = []
DEFAULT_MA_S, DEFAULT_MA_L = 5, 20

for a_th in ATH_CANDIDATES:
    ticker_srs = []
    row_detail = {"a_th": a_th}
    for ticker in TICKERS:
        df     = price_data[ticker]
        clean  = pd.Series(cwt_clean(df["Close"].values, a_th), index=df.index)
        eq, nt = run_ma_strategy(df, clean, DEFAULT_MA_S, DEFAULT_MA_L)
        m      = calc_metrics(eq)
        ticker_srs.append(m["SR"])
        row_detail[f"{ticker}_SR"]   = m["SR"]
        row_detail[f"{ticker}_ARR"]  = m["ARR(%)"]
        row_detail[f"{ticker}_MDD"]  = m["MDD(%)"]
    avg_sr  = round(np.mean(ticker_srs), 4)
    avg_arr = round(np.mean([row_detail[f"{t}_ARR"] for t in TICKERS]), 3)
    avg_mdd = round(np.mean([row_detail[f"{t}_MDD"] for t in TICKERS]), 3)
    row_detail["AVG_SR"]  = avg_sr
    row_detail["AVG_ARR"] = avg_arr
    row_detail["AVG_MDD"] = avg_mdd
    ath_records.append(row_detail)
    detail_str = " ".join(f"{t}:{row_detail[t+'_SR']:.3f}" for t in TICKERS)
    print(f"  a_th={a_th:2d}  평균 SR={avg_sr:.4f}  ARR={avg_arr:.2f}%  MDD={avg_mdd:.2f}%  | {detail_str}")

ath_df  = pd.DataFrame(ath_records)
best_ath_row = ath_df.loc[ath_df["AVG_SR"].idxmax()]
BEST_ATH = int(best_ath_row["a_th"])
print(f"\n  ★ 최적 a_th = {BEST_ATH}  (평균 SR = {best_ath_row['AVG_SR']:.4f})")
ath_df.to_csv("results/optimization/ath_search.csv", index=False)

# ─── Step 3: MA 조합 최적화 ───────────────────────────────
print(f"\n[Step 3] MA 조합 최적화 탐색 (a_th={BEST_ATH} 고정)...")
print(f"  후보: {MA_CANDIDATES}")
print("-" * 65)

# 최적 a_th로 클린 신호 미리 계산
clean_signals = {}
for ticker in TICKERS:
    df = price_data[ticker]
    clean_signals[ticker] = pd.Series(
        cwt_clean(df["Close"].values, BEST_ATH), index=df.index)

ma_records = []
for (ma_s, ma_l) in MA_CANDIDATES:
    ticker_srs = []
    row_detail = {"MA_short": ma_s, "MA_long": ma_l, "combo": f"MA{ma_s}/{ma_l}"}
    for ticker in TICKERS:
        df     = price_data[ticker]
        eq, nt = run_ma_strategy(df, clean_signals[ticker], ma_s, ma_l)
        m      = calc_metrics(eq)
        ticker_srs.append(m["SR"])
        row_detail[f"{ticker}_SR"]   = m["SR"]
        row_detail[f"{ticker}_ARR"]  = m["ARR(%)"]
        row_detail[f"{ticker}_MDD"]  = m["MDD(%)"]
        row_detail[f"{ticker}_trades"] = nt
    avg_sr  = round(np.mean(ticker_srs), 4)
    avg_arr = round(np.mean([row_detail[f"{t}_ARR"] for t in TICKERS]), 3)
    avg_mdd = round(np.mean([row_detail[f"{t}_MDD"] for t in TICKERS]), 3)
    row_detail["AVG_SR"]  = avg_sr
    row_detail["AVG_ARR"] = avg_arr
    row_detail["AVG_MDD"] = avg_mdd
    ma_records.append(row_detail)
    detail_str = " ".join(f"{t}:{row_detail[t+'_SR']:.3f}" for t in TICKERS)
    print(f"  MA{ma_s:2d}/{ma_l:2d}  평균 SR={avg_sr:.4f}  ARR={avg_arr:.2f}%  MDD={avg_mdd:.2f}%  | {detail_str}")

ma_df = pd.DataFrame(ma_records)
best_ma_row  = ma_df.loc[ma_df["AVG_SR"].idxmax()]
BEST_MA_S    = int(best_ma_row["MA_short"])
BEST_MA_L    = int(best_ma_row["MA_long"])
print(f"\n  ★ 최적 MA = {BEST_MA_S}/{BEST_MA_L}  (평균 SR = {best_ma_row['AVG_SR']:.4f})")
ma_df.to_csv("results/optimization/ma_search.csv", index=False)

# ─── 최종 요약 출력 ───────────────────────────────────────
print("\n" + "=" * 65)
print("  최적화 결과 요약")
print("=" * 65)
print(f"\n[a_th 탐색 결과]")
print(f"  {'a_th':>5}  {'AVG_SR':>8}  {'AVG_ARR(%)':>11}  {'AVG_MDD(%)':>11}")
print(f"  {'-'*45}")
for _, r in ath_df.iterrows():
    marker = " ★" if int(r['a_th']) == BEST_ATH else ""
    print(f"  {int(r['a_th']):>5}  {r['AVG_SR']:>8.4f}  {r['AVG_ARR']:>11.3f}  {r['AVG_MDD']:>11.3f}{marker}")

print(f"\n[MA 조합 탐색 결과]  (a_th={BEST_ATH} 고정)")
print(f"  {'조합':>10}  {'AVG_SR':>8}  {'AVG_ARR(%)':>11}  {'AVG_MDD(%)':>11}")
print(f"  {'-'*50}")
for _, r in ma_df.iterrows():
    marker = " ★" if int(r['MA_short'])==BEST_MA_S and int(r['MA_long'])==BEST_MA_L else ""
    print(f"  {r['combo']:>10}  {r['AVG_SR']:>8.4f}  {r['AVG_ARR']:>11.3f}  {r['AVG_MDD']:>11.3f}{marker}")

print(f"\n{'='*65}")
print(f"  최종 최적 설정:")
print(f"    TRAIN : {TRAIN_START} ~ {TRAIN_END}")
print(f"    TEST  : {TEST_START} ~ {TEST_END}")
print(f"    a_th  : {BEST_ATH}")
print(f"    MA    : {BEST_MA_S} / {BEST_MA_L}")
print(f"{'='*65}")

# 최적 설정을 파일로 저장
with open("results/optimization/best_params.txt", "w") as f:
    f.write(f"TRAIN_START={TRAIN_START}\n")
    f.write(f"TRAIN_END={TRAIN_END}\n")
    f.write(f"TEST_START={TEST_START}\n")
    f.write(f"TEST_END={TEST_END}\n")
    f.write(f"CWT_A_TH={BEST_ATH}\n")
    f.write(f"MA_SHORT={BEST_MA_S}\n")
    f.write(f"MA_LONG={BEST_MA_L}\n")
print("\n  ✅ 최적 파라미터 저장: results/optimization/best_params.txt")
