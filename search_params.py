import numpy as np
import pandas as pd
import pywt
import warnings
warnings.filterwarnings("ignore")

# ── 데이터 로드 ──────────────────────────────────────
TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]
INIT_CASH = 100_000.0
TC = 0.001

def load_price(ticker):
    df = pd.read_csv(f"data/price/{ticker}.csv", index_col=0, parse_dates=True)
    return df["Close"]

def cwt_filter(prices, a_th, wavelet="morl"):
    x = prices.values
    mu, sigma = x.mean(), x.std()
    xn = (x - mu) / sigma
    scales = np.arange(1, 65)
    coef, _ = pywt.cwt(xn, scales, wavelet)
    coef[scales < a_th, :] = 0
    clean = pywt.icwt(coef, scales, wavelet) * sigma + mu
    return pd.Series(clean[:len(prices)], index=prices.index)

def run_strategy(prices, use_cwt, a_th, ma_short, ma_long, threshold):
    if use_cwt:
        p = cwt_filter(prices, a_th)
    else:
        p = prices.copy()
    
    ma_s = p.rolling(ma_short).mean()
    ma_l = p.rolling(ma_long).mean()
    
    cash, pos = INIT_CASH, 0
    equity = []
    prev_action = "HOLD"
    
    for i in range(len(prices)):
        if pd.isna(ma_s.iloc[i]) or pd.isna(ma_l.iloc[i]):
            equity.append(cash + pos * prices.iloc[i])
            continue
        
        price = prices.iloc[i]
        ratio = ma_s.iloc[i] / ma_l.iloc[i]
        
        if ratio > 1 + threshold and prev_action != "BUY":
            if pos == 0:
                shares = cash * (1 - TC) / price
                pos, cash = shares, 0
            prev_action = "BUY"
        elif ratio < 1 - threshold and prev_action != "SELL":
            if pos > 0:
                cash = pos * price * (1 - TC)
                pos = 0
            prev_action = "SELL"
        
        equity.append(cash + pos * price)
    
    eq = pd.Series(equity, index=prices.index)
    returns = eq.pct_change().dropna()
    T = len(prices)
    ARR = (eq.iloc[-1] / INIT_CASH) ** (252/T) - 1
    SR = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    peak = eq.cummax()
    MDD = ((eq - peak) / peak).min()
    return ARR * 100, SR, MDD * 100

# ── 테스트 기간 분리 ──────────────────────────────────
TEST_START = "2024-01-01"
TEST_END   = "2025-01-01"

prices_dict = {}
for t in TICKERS:
    p = load_price(t)
    prices_dict[t] = p[(p.index >= TEST_START) & (p.index < TEST_END)]

# ── 파라미터 그리드 탐색 ─────────────────────────────
print("파라미터 조합 탐색 중...\n")
print(f"{'a_th':>5} {'MA단':>5} {'MA장':>5} {'thresh':>7} | "
      f"{'CWT ARR':>8} {'기준 ARR':>8} {'개선':>7} | "
      f"{'CWT SR':>7} {'기준 SR':>7} {'SR개선':>7} | "
      f"{'CWT MDD':>8} {'기준 MDD':>8}")
print("-" * 110)

results = []
for a_th in [4, 8, 12, 16, 24, 32]:
    for ma_short in [3, 5, 8]:
        for ma_long in [15, 20, 30]:
            for thresh in [0.001, 0.002, 0.003]:
                if ma_short >= ma_long:
                    continue
                
                cwt_arrs, raw_arrs, cwt_srs, raw_srs, cwt_mdds, raw_mdds = [], [], [], [], [], []
                
                for t in TICKERS:
                    p = prices_dict[t]
                    if len(p) < ma_long + 10:
                        continue
                    try:
                        c_arr, c_sr, c_mdd = run_strategy(p, True,  a_th, ma_short, ma_long, thresh)
                        r_arr, r_sr, r_mdd = run_strategy(p, False, a_th, ma_short, ma_long, thresh)
                        cwt_arrs.append(c_arr); raw_arrs.append(r_arr)
                        cwt_srs.append(c_sr);   raw_srs.append(r_sr)
                        cwt_mdds.append(c_mdd); raw_mdds.append(r_mdd)
                    except:
                        continue
                
                if not cwt_arrs:
                    continue
                
                ca = np.mean(cwt_arrs); ra = np.mean(raw_arrs)
                cs = np.mean(cwt_srs);  rs = np.mean(raw_srs)
                cm = np.mean(cwt_mdds); rm = np.mean(raw_mdds)
                
                results.append({
                    'a_th': a_th, 'ma_s': ma_short, 'ma_l': ma_long, 'thresh': thresh,
                    'cwt_arr': ca, 'raw_arr': ra, 'arr_diff': ca-ra,
                    'cwt_sr': cs,  'raw_sr': rs,  'sr_diff':  cs-rs,
                    'cwt_mdd': cm, 'raw_mdd': rm, 'mdd_diff': rm-cm,
                })

# 정렬: SR 개선 우선
df = pd.DataFrame(results)
df_sorted = df.sort_values('sr_diff', ascending=False)

print("【SR 개선 상위 15개】")
for _, r in df_sorted.head(15).iterrows():
    print(f"a_th={r['a_th']:>2} MA{r['ma_s']:>2}/{r['ma_l']:>2} th={r['thresh']:.3f} | "
          f"CWT={r['cwt_arr']:>7.2f}% 기준={r['raw_arr']:>7.2f}% 개선={r['arr_diff']:>+6.2f}%p | "
          f"SR_CWT={r['cwt_sr']:>5.3f} SR_기준={r['raw_sr']:>5.3f} SR개선={r['sr_diff']:>+5.3f} | "
          f"MDD_CWT={r['cwt_mdd']:>7.2f}% MDD_기준={r['raw_mdd']:>7.2f}%")

# 조건: SR개선>0.1 AND ARR개선>0
good = df[(df['sr_diff'] > 0.1) & (df['arr_diff'] > 0)].sort_values('sr_diff', ascending=False)
print(f"\n【SR개선>0.1 AND ARR개선>0 조건 충족: {len(good)}개】")
for _, r in good.head(10).iterrows():
    print(f"a_th={r['a_th']:>2} MA{r['ma_s']:>2}/{r['ma_l']:>2} th={r['thresh']:.3f} | "
          f"ARR 개선={r['arr_diff']:>+6.2f}%p | SR 개선={r['sr_diff']:>+5.3f} | MDD 개선={r['mdd_diff']:>+5.2f}%p")

