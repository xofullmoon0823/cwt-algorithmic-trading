# ============================================================
# 환경 세팅 완전 가이드 + 실험 실행기
# CWT 기반 Timeliness-Aware 금융 에이전트 프로젝트
# 변문성 (20223872)
#
# OpenAI API 없음 → Ollama (로컬 무료 LLM) 사용
# PC 로컬 환경 기준
# ============================================================

"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 0 — 이 파일 실행 전에 PC에서 먼저 할 것 (1회만)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1] 프로젝트 폴더 만들기
    - 바탕화면이나 원하는 위치에 폴더 하나 만들기
    - 예: C:/Users/변문성/finagent_cwt/
    - 이 파이썬 파일을 그 폴더에 저장

[2] VS Code 터미널 열기
    - VS Code에서 폴더 열고 → 상단 메뉴 Terminal → New Terminal

[3] 파이썬 패키지 설치 (터미널에 복붙해서 실행)

    pip install yfinance pywavelets pandas numpy matplotlib seaborn
    pip install requests tqdm scikit-learn

[4] Ollama 설치 (LLM 로컬 실행용 — GPT-4 대체)

    ① https://ollama.com 접속 → Download 클릭 → Windows/Mac 설치
    ② 설치 완료 후 터미널에서:

       RAM 8GB 이하 (현재 환경):  ollama pull llama3.1:8b   ← 이걸로 실행
       RAM 16GB:                  ollama pull llama3.1:8b
       RAM 32GB 이상:             ollama pull llama3.1:70b

    ③ 설치 확인:
       ollama run llama3.1:8b "hello"
       → 응답이 뜨면 성공

[5] Finnhub API키 발급 (뉴스 데이터용 — 무료)
    ① https://finnhub.io 접속 → 우측 상단 Get free API key
    ② 구글 계정으로 30초 가입
    ③ 대시보드에서 API key 복사
    ④ 아래 FINNHUB_API_KEY 변수에 붙여넣기

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 설치 완료 후 이 파일을 VS Code에서 실행 (F5 또는 Run)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 여기만 본인 설정으로 바꾸면 됩니다
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINNHUB_API_KEY = "여기에_핀허브_API키_붙여넣기"   # finnhub.io 무료 발급
OLLAMA_MODEL    = "llama3.1:8b"                   # 8GB 이하 PC 최적화 (가장 가벼운 버전)
PROJECT_DIR     = "."                              # 현재 폴더 기준 (그대로 두세요)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pywt
import requests
from tqdm import tqdm
warnings.filterwarnings("ignore")

# ── 폴더 구조 자동 생성 ─────────────────────────
for d in ["data/price", "data/news", "data/clean",
          "results/cwt", "results/baselines",
          "results/agent", "results/paper_tables"]:
    os.makedirs(os.path.join(PROJECT_DIR, d), exist_ok=True)

print("=" * 58)
print("  CWT-FinAgent 프로젝트 시작")
print("=" * 58)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 1: 환경 체크
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def check_environment():
    print("\n[1/4] 환경 체크 중...")
    results = {}

    # 패키지 확인
    packages = {
        "yfinance":    "yfinance",
        "pywt":        "PyWavelets",
        "pandas":      "pandas",
        "numpy":       "numpy",
        "matplotlib":  "matplotlib",
        "requests":    "requests",
        "sklearn":     "scikit-learn",
    }
    all_ok = True
    for mod, pip_name in packages.items():
        try:
            __import__(mod)
            print(f"  ✅ {pip_name}")
        except ImportError:
            print(f"  ❌ {pip_name} 없음 → pip install {pip_name}")
            all_ok = False
    results["packages_ok"] = all_ok

    # Ollama 확인
    print("\n  Ollama 확인 중...")
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        if models:
            print(f"  ✅ Ollama 실행 중 | 설치된 모델: {models}")
            results["ollama_ok"] = True
            results["ollama_models"] = models
        else:
            print(f"  ⚠️  Ollama 실행 중이지만 모델 없음")
            print(f"     터미널에서 실행: ollama pull {OLLAMA_MODEL}")
            results["ollama_ok"] = False
    except:
        print("  ⚠️  Ollama 미실행 — 지금 당장 없어도 됩니다")
        print("     Step 1~3 (데이터 수집, CWT, Baseline)은 Ollama 없이 실행 가능")
        print(f"     LLM 실험(Step 4) 전까지만 설치하면 됩니다")
        results["ollama_ok"] = False

    # Finnhub 키 확인
    if FINNHUB_API_KEY != "여기에_핀허브_API키_붙여넣기":
        print(f"\n  ✅ Finnhub API키 입력됨")
        results["finnhub_ok"] = True
    else:
        print(f"\n  ⚠️  Finnhub API키 미입력")
        print(f"     뉴스 없이도 주가 데이터만으로 CWT 실험은 진행 가능합니다")
        results["finnhub_ok"] = False

    print(f"\n  패키지: {'OK' if results['packages_ok'] else '일부 설치 필요'}")
    return results


env_status = check_environment()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 2: 주가 데이터 수집
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TICKERS     = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]
TRAIN_START = "2022-06-01"
TRAIN_END   = "2023-06-01"
TEST_START  = "2023-06-01"
TEST_END    = "2024-01-01"
INIT_CASH   = 100_000

def collect_price_data():
    import yfinance as yf

    print("\n[2/4] 주가 데이터 수집 중...")
    price_data = {}

    for ticker in TICKERS:
        save_path = f"data/price/{ticker}_price.csv"

        # 이미 수집한 경우 스킵
        if os.path.exists(save_path):
            df = pd.read_csv(save_path, index_col="Date", parse_dates=True)
            price_data[ticker] = df
            print(f"  ⏩ {ticker}: 기존 파일 로드 ({len(df)}일)")
            continue

        try:
            df = yf.download(ticker, start=TRAIN_START, end=TEST_END,
                             interval="1d", progress=False, auto_adjust=True)
            df.columns = [c[0] if isinstance(c, tuple) else c
                          for c in df.columns]
            df = df[["Open","High","Low","Close","Volume"]].dropna()
            df.index.name = "Date"
            df.to_csv(save_path)
            price_data[ticker] = df
            print(f"  ✅ {ticker}: {len(df)}거래일 수집 완료")
        except Exception as e:
            print(f"  ❌ {ticker} 실패: {e}")

    print(f"\n  총 {len(price_data)}개 종목 준비 완료")
    return price_data


price_data = collect_price_data()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 3: CWT 필터 + 클린 신호 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CWTFilter:
    """
    논문 핵심 모듈
    수식(1): W_x(a,b) = (1/√|a|) ∫ x(t)·ψ*((t-b)/a) dt
    수식(5): x̂(t) ≈ (1/N_a) · Σ_{a≥a_th} W_x(a,t)
    """
    def __init__(self, wavelet="morl", scale_threshold=8):
        self.wavelet         = wavelet
        self.scales          = np.arange(1, 65)
        self.scale_threshold = scale_threshold

    def transform(self, close: np.ndarray) -> dict:
        mu, sigma   = close.mean(), close.std() or 1.0
        x_norm      = (close - mu) / sigma

        coef, freqs = pywt.cwt(x_norm, self.scales, self.wavelet)

        coef_clean  = coef.copy()
        mask        = self.scales < self.scale_threshold
        coef_clean[mask, :] = 0

        n_low = (~mask).sum()
        x_clean_norm = (np.sum(coef_clean, axis=0) / n_low
                        if n_low > 0 else x_norm)
        x_clean = x_clean_norm * sigma + mu
        x_noise = close - x_clean

        noise_power  = np.mean(x_noise**2)
        signal_power = np.mean(x_clean**2) + 1e-10
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

        return {
            "clean":       x_clean,
            "noise":       x_noise,
            "coef":        coef,
            "coef_clean":  coef_clean,
            "snr":         snr,
            "noise_ratio": noise_power / (signal_power + noise_power),
        }


def generate_clean_signals(price_data, scale_threshold=8):
    print(f"\n[3/4] CWT 클린 신호 생성 중... (a_th={scale_threshold})")
    cwt  = CWTFilter(scale_threshold=scale_threshold)
    clean_data = {}

    for ticker, df in price_data.items():
        save_path = f"data/clean/{ticker}_clean_ath{scale_threshold}.csv"
        close     = df["Close"].values
        res       = cwt.transform(close)

        out_df = df.copy()
        out_df["Close_clean"] = res["clean"]
        out_df["Noise"]       = res["noise"]
        out_df.to_csv(save_path)
        clean_data[ticker] = {"df": out_df, "result": res}

        print(f"  ✅ {ticker:6s}  SNR={res['snr']:6.1f}dB  "
              f"노이즈={res['noise_ratio']*100:.1f}%")

    return clean_data


clean_data = generate_clean_signals(price_data, scale_threshold=8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 4: 성능 지표 계산 (FinBen 6개 지표)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def calc_metrics(equity: pd.Series, rfr=0.04) -> dict:
    """ARR / SR / CR / SOR / MDD / VOL"""
    ret = equity.pct_change().dropna()
    n   = len(ret)
    if n == 0:
        return {k: 0 for k in ["ARR(%)","SR","CR","SOR","MDD(%)","VOL(%)"]}

    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    ARR = (1 + total_ret) ** (252 / n) - 1
    VOL = ret.std() * (252 ** 0.5)
    excess = ret.mean() * 252 - rfr
    SR  = excess / VOL if VOL > 0 else 0
    rolling_max = equity.cummax()
    MDD = ((equity - rolling_max) / rolling_max).min()
    CR  = ARR / abs(MDD) if MDD else 0
    downside_vol = ret[ret < 0].std() * (252 ** 0.5) if (ret < 0).any() else 1e-8
    SOR = excess / downside_vol

    return {
        "ARR(%)": round(ARR * 100, 2),
        "SR":     round(SR, 3),
        "CR":     round(CR, 3),
        "SOR":    round(SOR, 3),
        "MDD(%)": round(MDD * 100, 2),
        "VOL(%)": round(VOL * 100, 2),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 5: Baseline 전략 실험
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_bah(df):
    test  = df.loc[TEST_START:TEST_END]
    shares = INIT_CASH / test["Close"].iloc[0]
    return pd.Series(test["Close"].values * shares, index=test.index)

def run_macd(df, fast=12, slow=26, sig=9):
    d = df.copy()
    d["macd"]   = (d["Close"].ewm(span=fast, adjust=False).mean()
                 - d["Close"].ewm(span=slow, adjust=False).mean())
    d["signal"] = d["macd"].ewm(span=sig, adjust=False).mean()
    d["cross"]  = np.sign(d["macd"] - d["signal"]).diff()
    test = d.loc[TEST_START:TEST_END]
    cash, pos = float(INIT_CASH), 0.0
    eq = []
    for _, row in test.iterrows():
        p = float(row["Close"])
        if row["cross"] > 0 and cash > 0:
            pos, cash = cash / p, 0.0
        elif row["cross"] < 0 and pos > 0:
            cash, pos = pos * p, 0.0
        eq.append(cash + pos * p)
    return pd.Series(eq, index=test.index)

def run_rsi(df, period=14, ob=70, os_=30):
    d     = df.copy()
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(com=period-1, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-10))
    test = d.loc[TEST_START:TEST_END]
    cash, pos, in_pos = float(INIT_CASH), 0.0, False
    eq = []
    for _, row in test.iterrows():
        p = float(row["Close"])
        if row["rsi"] < os_ and not in_pos and cash > 0:
            pos, cash, in_pos = cash / p, 0.0, True
        elif row["rsi"] > ob and in_pos and pos > 0:
            cash, pos, in_pos = pos * p, 0.0, False
        eq.append(cash + pos * p)
    return pd.Series(eq, index=test.index)


print("\n[4/4] Baseline 전략 실험 중...")
baseline_records = []

for ticker, df in price_data.items():
    for name, func in [("B&H", run_bah), ("MACD", run_macd), ("RSI", run_rsi)]:
        try:
            equity = func(df)
            m = calc_metrics(equity)
            m.update({"Ticker": ticker, "Strategy": name})
            baseline_records.append(m)
            equity.to_csv(f"results/baselines/{ticker}_{name}_equity.csv",
                          header=["Equity"])
        except Exception as e:
            print(f"  ❌ {ticker}/{name}: {e}")

baseline_df = pd.DataFrame(baseline_records)
cols = ["Ticker","Strategy","ARR(%)","SR","CR","SOR","MDD(%)","VOL(%)"]
baseline_df = baseline_df[cols]
baseline_df.to_csv("results/baselines/baseline_results.csv", index=False)

print("\n  📊 Baseline 결과 요약")
print("  " + "-"*60)
avg = baseline_df.groupby("Strategy")[["ARR(%)","SR","MDD(%)","VOL(%)"]].mean().round(2)
print(avg.to_string())
print("  " + "-"*60)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 6: Ollama LLM 에이전트 (GPT-4 대체)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class OllamaAgent:
    """
    Ollama 로컬 LLM을 백본으로 사용하는 금융 에이전트
    GPT-4-Turbo 대체 — 완전 무료, 로컬 실행
    """
    def __init__(self, model=OLLAMA_MODEL,
                 base_url="http://localhost:11434"):
        self.model    = model
        self.base_url = base_url
        self.history  = []   # 반성 모듈용 히스토리

    def _call(self, prompt: str, temperature=0.3) -> str:
        """Ollama API 호출"""
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model,
                      "prompt": prompt,
                      "stream": False,
                      "options": {"temperature": temperature}},
                timeout=120
            )
            return r.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            return "OLLAMA_NOT_RUNNING"
        except Exception as e:
            return f"ERROR:{e}"

    def analyze_market(self, ticker: str, clean_price: float,
                       raw_price: float, news_headlines: list,
                       recent_prices: list) -> dict:
        """시장 분석 → 매매 결정"""
        import re as _re

        if len(recent_prices) >= 5:
            trend_5d  = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5] * 100
            trend_20d = (recent_prices[-1] - recent_prices[-20]) / recent_prices[-20] * 100                         if len(recent_prices) >= 20 else 0.0
        else:
            trend_5d = trend_20d = 0.0

        noise_pct = abs(raw_price - clean_price) / (raw_price + 1e-10) * 100

        prompt = f"""You are a stock trader for {ticker}.
Data:
- Clean price: ${clean_price:.2f}
- Raw price: ${raw_price:.2f}
- Noise: {noise_pct:.1f}%
- 5-day trend: {trend_5d:+.1f}%
- 20-day trend: {trend_20d:+.1f}%
History: {self._get_reflection_summary()}

Rules: trend>+1.5% AND noise<4% → BUY, trend<-1.5% → SELL, else → HOLD
Reply ONE word only (BUY or SELL or HOLD):"""

        response = self._call(prompt, temperature=0.1).strip().upper()
        words = _re.findall(r'\b(BUY|SELL|HOLD)\b', response)

        if words:
            decision = words[0]
        elif trend_5d > 1.5 and noise_pct < 4:
            decision = "BUY"
        elif trend_5d < -1.5:
            decision = "SELL"
        else:
            decision = "HOLD"

        return {
            "decision": decision,
            "confidence": 0.7,
            "reason": f"5d={trend_5d:+.1f}% noise={noise_pct:.1f}%",
            "signal_quality": "noisy" if noise_pct > 4 else "good"
        }

    def reflect(self, ticker: str, decision: str, outcome_pct: float):
        """
        거래 결과 반성 모듈 (FinAgent dual-level reflection 대응)
        저수준 반성: 단기 결과 기록
        """
        self.history.append({
            "ticker":      ticker,
            "decision":    decision,
            "outcome_pct": outcome_pct,
            "timestamp":   time.strftime("%Y-%m-%d")
        })
        # 히스토리 최대 20개 유지
        if len(self.history) > 20:
            self.history.pop(0)

    def _get_reflection_summary(self) -> str:
        if not self.history:
            return "No previous trades."
        recent = self.history[-3:]
        lines  = []
        for h in recent:
            lines.append(f"  [{h['timestamp']}] {h['ticker']} {h['decision']} "
                         f"→ {h['outcome_pct']:+.1f}%")
        return "\n".join(lines)

    def is_available(self) -> bool:
        test = self._call("say ok")
        return test not in ["OLLAMA_NOT_RUNNING"] and not test.startswith("ERROR")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 7: CWT-FinAgent 실험 (Ollama 기반)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_cwt_agent(ticker: str, use_cwt: bool = True,
                  scale_threshold: int = 8,
                  agent: OllamaAgent = None) -> pd.Series:
    """
    CWT-FinAgent (use_cwt=True) 또는 FinAgent 원본 (use_cwt=False) 실험

    Parameters
    ----------
    ticker         : 종목 코드
    use_cwt        : True → CWT 클린 신호 사용 (제안 방법)
                     False → 원 신호 사용 (FinAgent 원본 재현)
    scale_threshold: CWT 스케일 임계값
    agent          : OllamaAgent 인스턴스
    """
    price_path = f"data/price/{ticker}_price.csv"
    if not os.path.exists(price_path):
        print(f"  ❌ {ticker}: 주가 데이터 없음")
        return pd.Series(dtype=float)

    df     = pd.read_csv(price_path, index_col="Date", parse_dates=True)
    test   = df.loc[TEST_START:TEST_END].copy()

    # CWT 클린 신호 준비
    cwt    = CWTFilter(scale_threshold=scale_threshold)
    full_close = df["Close"].values
    res    = cwt.transform(full_close)
    clean_full = pd.Series(res["clean"], index=df.index)

    # 뉴스 데이터 로드 (있으면 사용)
    news_df = pd.DataFrame()
    news_path = f"data/news/{ticker}_news.csv"
    if os.path.exists(news_path):
        try:
            news_df = pd.read_csv(news_path, parse_dates=["datetime"])
        except:
            pass

    # 에이전트 없는 경우 룰 기반으로 대체
    use_llm = (agent is not None and agent.is_available())
    if not use_llm:
        print(f"  ⚠️  {ticker}: Ollama 미실행 → 룰 기반 에이전트로 대체")

    cash, holdings = float(INIT_CASH), 0.0
    equity_list    = []
    all_prices     = df["Close"].values
    test_start_idx = list(df.index).index(test.index[0])

    print(f"  📈 {ticker} {'CWT-FinAgent' if use_cwt else 'FinAgent(원본)'} "
          f"실험 중 ({len(test)}일)...", end="", flush=True)

    for step, (date, row) in enumerate(test.iterrows()):
        raw_price   = float(row["Close"])
        clean_price = float(clean_full.loc[date]) if date in clean_full.index \
                      else raw_price

        # 에이전트 입력 가격 결정
        input_price = clean_price if use_cwt else raw_price

        # 최근 30일 가격 (에이전트 컨텍스트)
        idx = test_start_idx + step
        recent_raw   = list(all_prices[max(0, idx-30):idx+1])
        recent_clean = list(clean_full.values[max(0, idx-30):idx+1])
        recent_input = recent_clean if use_cwt else recent_raw

        # 당일 뉴스 헤드라인
        headlines = []
        if not news_df.empty:
            day_news  = news_df[news_df["datetime"].dt.date == date.date()]
            headlines = day_news["headline"].tolist()[:5]

        # ── 매매 결정 ──────────────────────────────────
        if use_llm:
            result = agent.analyze_market(
                ticker, input_price, raw_price,
                headlines, recent_input
            )
            decision = result.get("decision", "HOLD")
        else:
            # 룰 기반 대체 (LLM 없을 때)
            # 5일 이동평균 돌파 + 노이즈 레벨 필터
            if len(recent_input) >= 5:
                ma5 = np.mean(recent_input[-5:])
                noise_level = abs(raw_price - clean_price) / (raw_price + 1e-10)
                if input_price > ma5 * 1.01 and noise_level < 0.03:
                    decision = "BUY"
                elif input_price < ma5 * 0.99:
                    decision = "SELL"
                else:
                    decision = "HOLD"
            else:
                decision = "HOLD"

        # ── 주문 집행 ──────────────────────────────────
        if decision == "BUY" and cash > 100:
            holdings = cash / raw_price   # 실제 체결은 raw price
            cash     = 0.0
        elif decision == "SELL" and holdings > 0:
            cash     = holdings * raw_price
            holdings = 0.0

        current_equity = cash + holdings * raw_price
        equity_list.append(current_equity)

        # 반성 업데이트 (5거래일마다)
        if use_llm and step > 0 and step % 5 == 0:
            prev_equity = equity_list[-6] if len(equity_list) > 5 else INIT_CASH
            pct_change  = (current_equity - prev_equity) / prev_equity * 100
            agent.reflect(ticker, decision, pct_change)

    print(" 완료")
    return pd.Series(equity_list, index=test.index)


# ── 실험 실행 ──────────────────────────────────────
print("\n[Agent 실험] Ollama 연결 확인 중...")
agent = OllamaAgent(model=OLLAMA_MODEL)
ollama_ok = agent.is_available()

if ollama_ok:
    print(f"  ✅ Ollama 연결 성공 ({OLLAMA_MODEL})")
else:
    print(f"  ⚠️  Ollama 미연결 → 룰 기반 에이전트로 실험 진행")
    print(f"     나중에 ollama pull {OLLAMA_MODEL} 실행 후 재실행 가능")

agent_records = []
print("\n  CWT-FinAgent vs FinAgent 실험 시작...")

for ticker in TICKERS:
    for use_cwt, label in [(False, "FinAgent(원본)"),
                           (True,  "CWT-FinAgent(제안)")]:
        equity = run_cwt_agent(
            ticker, use_cwt=use_cwt,
            scale_threshold=8,
            agent=agent if ollama_ok else None
        )
        if equity.empty:
            continue
        m = calc_metrics(equity)
        m.update({"Ticker": ticker, "Strategy": label})
        agent_records.append(m)
        equity.to_csv(
            f"results/agent/{ticker}_{label.replace('(','').replace(')','').replace(' ','_')}_equity.csv",
            header=["Equity"]
        )

agent_df = pd.DataFrame(agent_records) if agent_records else pd.DataFrame()
if not agent_df.empty:
    cols = ["Ticker","Strategy","ARR(%)","SR","CR","SOR","MDD(%)","VOL(%)"]
    agent_df = agent_df[[c for c in cols if c in agent_df.columns]]
    agent_df.to_csv("results/agent/agent_results.csv", index=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 8: 논문 결과 테이블 자동 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def make_paper_table():
    """
    논문 4장 결과 테이블을 CSV로 자동 생성
    → 이 CSV를 Word 논문에 그대로 붙여넣기
    """
    frames = []
    if not baseline_df.empty:
        frames.append(baseline_df)
    if not agent_df.empty:
        frames.append(agent_df)

    if not frames:
        print("\n  결과 데이터 없음")
        return

    all_df = pd.concat(frames, ignore_index=True)
    cols   = ["Ticker","Strategy","ARR(%)","SR","CR","SOR","MDD(%)","VOL(%)"]
    all_df = all_df[[c for c in cols if c in all_df.columns]]

    # 종목별 테이블
    for ticker in TICKERS:
        sub = all_df[all_df["Ticker"] == ticker]
        if sub.empty:
            continue
        save = f"results/paper_tables/{ticker}_results_table.csv"
        sub.drop(columns="Ticker").to_csv(save, index=False)

    # 전체 평균 테이블
    avg_all = all_df.groupby("Strategy")[
        ["ARR(%)","SR","CR","SOR","MDD(%)","VOL(%)"]
    ].mean().round(3)
    avg_all.to_csv("results/paper_tables/AVERAGE_results_table.csv")

    print("\n  📋 전체 전략 평균 성능 (논문 4장 표)")
    print("  " + "="*65)
    print("  " + avg_all.to_string().replace("\n", "\n  "))
    print("  " + "="*65)

    # CWT-FinAgent vs FinAgent 개선율 계산
    if "CWT-FinAgent(제안)" in avg_all.index and "FinAgent(원본)" in avg_all.index:
        cwt_arr  = avg_all.loc["CWT-FinAgent(제안)","ARR(%)"]
        base_arr = avg_all.loc["FinAgent(원본)","ARR(%)"]
        if base_arr != 0:
            improvement = (cwt_arr - base_arr) / abs(base_arr) * 100
            print(f"\n  🎯 CWT-FinAgent ARR 개선율: {improvement:+.1f}%"
                  f" (FinAgent 원본 대비)")

    print(f"\n  ✅ 논문용 테이블 저장: results/paper_tables/")

make_paper_table()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 9: 핵심 그래프 자동 생성 (논문용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def make_paper_figures():
    """논문에 들어갈 핵심 그래프 3개 자동 생성"""

    # ── 그림 1: CWT 필터 전후 비교 (AAPL, 논문 3장) ──
    if "AAPL" in clean_data:
        aapl_df  = clean_data["AAPL"]["df"]
        test     = aapl_df.loc[TEST_START:TEST_END]
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

        axes[0].plot(test.index, test["Close"],       color="#AAAAAA",
                     lw=0.9, label="원 신호 (Raw)")
        axes[0].plot(test.index, test["Close_clean"], color="#2E75B6",
                     lw=1.6, label="CWT 클린 신호")
        axes[0].set_title("AAPL — CWT 필터 전/후 가격 신호 비교 (테스트 기간)",
                          fontsize=13, fontweight="bold")
        axes[0].set_ylabel("Price (USD)")
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        axes[1].fill_between(test.index, test["Noise"], 0,
                             where=test["Noise"] >= 0,
                             color="#C55A11", alpha=0.6, label="양(+) 노이즈")
        axes[1].fill_between(test.index, test["Noise"], 0,
                             where=test["Noise"] < 0,
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
        print("  ✅ 그림 1 저장: Fig1_cwt_before_after_AAPL.png  (논문 3장용)")

    # ── 그림 2: Ablation — a_th별 비교 (논문 5장) ────
    if "AAPL" in price_data:
        close = price_data["AAPL"]["Close"].values
        dates = price_data["AAPL"].loc[TEST_START:TEST_END].index
        thresholds = [4, 8, 16, 32]
        colors_ab  = ["#2E75B6","#C55A11","#1E6B3C","#7030A0"]

        fig, ax = plt.subplots(figsize=(13, 6))
        close_test = price_data["AAPL"].loc[TEST_START:TEST_END]["Close"].values
        ax.plot(dates, close_test, color="#CCCCCC", lw=0.8,
                alpha=0.8, label="원 신호")

        for a_th, col in zip(thresholds, colors_ab):
            cwt_ab  = CWTFilter(scale_threshold=a_th)
            res_ab  = cwt_ab.transform(price_data["AAPL"]["Close"].values)
            clean_s = pd.Series(res_ab["clean"],
                                index=price_data["AAPL"].index)
            clean_t = clean_s.loc[TEST_START:TEST_END]
            ax.plot(clean_t.index, clean_t.values, color=col, lw=1.4,
                    label=f"a_th={a_th}  (노이즈 {res_ab['noise_ratio']*100:.0f}% 제거)")

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
        print("  ✅ 그림 2 저장: Fig2_ablation_threshold_AAPL.png  (논문 5장용)")

    # ── 그림 3: 에쿼티 커브 비교 (논문 4장) ──────────
    eq_files = {
        "B&H":             "results/baselines/AAPL_B&H_equity.csv",
        "MACD":            "results/baselines/AAPL_MACD_equity.csv",
        "FinAgent(원본)":  "results/agent/AAPL_FinAgent원본_equity.csv",
        "CWT-FinAgent":    "results/agent/AAPL_CWT-FinAgent제안_equity.csv",
    }
    colors_eq = {
        "B&H":            "#888888",
        "MACD":           "#C55A11",
        "FinAgent(원본)": "#1E6B3C",
        "CWT-FinAgent":   "#2E75B6",
    }
    fig, ax = plt.subplots(figsize=(13, 6))
    plotted = False
    for label, path in eq_files.items():
        # 경로 패턴 유연하게 탐색
        candidates = [path]
        for f in os.listdir(os.path.dirname(path) or "."):
            if "AAPL" in f and label.replace("(","").replace(")","").replace(" ","_")[:6] in f:
                candidates.append(os.path.join(os.path.dirname(path), f))
        for c in candidates:
            if os.path.exists(c):
                eq = pd.read_csv(c, index_col=0, parse_dates=True).iloc[:,0]
                ax.plot(eq.index, eq.values,
                        color=colors_eq.get(label,"#2E75B6"),
                        lw=2.0 if "CWT" in label else 1.3,
                        linestyle="-" if "CWT" in label else "--",
                        label=label)
                plotted = True
                break

    if plotted:
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
        plt.savefig("results/paper_tables/Fig3_equity_curves_AAPL.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✅ 그림 3 저장: Fig3_equity_curves_AAPL.png  (논문 4장용)")


print("\n  논문용 그래프 생성 중...")
make_paper_figures()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 완료 요약
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "="*58)
print("  ✅ 전체 실험 완료!")
print("="*58)
print("""
생성된 파일 구조:
  data/
    price/      → 종목별 원 주가 CSV
    clean/      → CWT 클린 신호 CSV
    news/       → 뉴스 헤드라인 CSV (API키 입력 시)

  results/
    cwt/
      Fig1_cwt_before_after_AAPL.png   ← 논문 3장 그림
      Fig2_ablation_threshold_AAPL.png ← 논문 5장 그림
    baselines/
      baseline_results.csv             ← B&H / MACD / RSI 수치
    agent/
      agent_results.csv                ← FinAgent / CWT-FinAgent 수치
    paper_tables/
      AVERAGE_results_table.csv        ← 논문 4장 결과 표 (평균)
      AAPL_results_table.csv           ← 종목별 결과 표
      Fig3_equity_curves_AAPL.png      ← 논문 4장 그림

다음 할 일:
  1. results/paper_tables/AVERAGE_results_table.csv 열어서
     논문 초안 4장 결과 표에 수치 붙여넣기
  2. Fig1~3 그림을 논문 Word 파일에 삽입
  3. Ollama 설치 후 재실행 → LLM 에이전트 결과로 교체
""")
