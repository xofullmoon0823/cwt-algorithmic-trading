# CWT 기반 신호 전처리를 활용한 금융 자동매매 전략의 성능 향상
# Enhancing Financial Algorithmic Trading Performance via CWT-Based Signal Preprocessing

---

## 개요 / Overview

**[한국어]**  
연속 웨이블릿 변환(CWT)을 활용한 신호 전처리 기법을 금융 자동매매 전략에 적용하여 성능을 향상시키는 연구입니다. 이동평균(MA) 기반 매매 신호에 CWT 필터링을 결합하여 노이즈를 제거하고 추세 감지 정확도를 높입니다.

**[English]**  
This research applies Continuous Wavelet Transform (CWT)-based signal preprocessing to financial algorithmic trading strategies to improve performance. By combining CWT filtering with Moving Average (MA)-based trading signals, the system reduces noise and enhances trend detection accuracy.

---

## 저자 / Authors

| 역할 / Role | 이름 / Name |
|---|---|
| 저자 / Author | 변문성 (Byeon Munseong) |
| 지도교수 / Supervisor | 양진홍 교수 (Prof. Yang Jinhong) |

---

## 실험 설정 / Experimental Setup

| 항목 / Item | 내용 / Details |
|---|---|
| 학습 기간 / Training Period | 2019 – 2022 |
| 테스트 기간 / Test Period | 2023 – 2025 |
| CWT 임계값 / CWT Threshold (a_th) | 24 |
| 이동평균 / Moving Averages | MA3 (단기 / Short), MA15 (장기 / Long) |
| 대상 종목 / Target Stocks | AAPL, AMZN, GOOGL, MSFT, TSLA |

---

## 핵심 결과 / Key Results

| 지표 / Metric | 값 / Value |
|---|---|
| 연평균 수익률 / ARR (Annualized Return Rate) | **50.32%** |
| 샤프 비율 / Sharpe Ratio (SR) | **1.542** |
| 최대 낙폭 / MDD (Maximum Drawdown) | **-17.36%** |
| 연간 거래 횟수 / Trades per Year | **9.7회** |

---

## 프로젝트 구조 / Project Structure

```
finagent_cwt/
├── run_all_v4.py        # 메인 실행 스크립트 / Main execution script
├── run_all_v3.py        # 이전 버전 / Previous version
├── run_all.py           # 초기 버전 / Initial version
├── experiments.py       # 실험 모듈 / Experiment module
├── optimize_params.py   # 파라미터 최적화 / Parameter optimization
├── search_params.py     # 파라미터 탐색 / Parameter search
├── make_figures.py      # 결과 시각화 / Result visualization
├── fix_figures.py       # 도표 수정 / Figure correction
├── regen_figures.py     # 도표 재생성 / Figure regeneration
├── run_experiments.py   # 실험 실행기 / Experiment runner
├── data/                # 주가 데이터 / Stock price data
│   ├── price/           # 가격 데이터 / Price data
│   └── ...
├── results/             # 실험 결과 / Experiment results
│   ├── figures/         # 시각화 결과 / Visualization outputs
│   ├── logs/            # 실행 로그 / Execution logs
│   ├── paper_tables/    # 논문용 표 / Paper tables
│   └── ...
└── fixed_figs/          # 수정된 도표 / Fixed figures
```

---

## 실행 방법 / How to Run

### 1. 환경 설치 / Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. 메인 실험 실행 / Run Main Experiment

```bash
python3 run_all_v4.py
```

결과는 `results/` 디렉토리에 저장됩니다.  
Results are saved in the `results/` directory.

---

## 환경 / Environment

| 항목 / Item | 버전 / Version |
|---|---|
| Python | 3.13 |
| PyWavelets | 1.9 |
| 기타 / Others | `requirements.txt` 참조 / See `requirements.txt` |

---

## 방법론 개요 / Methodology Overview

**[한국어]**  
1. **데이터 수집**: yfinance를 통해 5개 미국 주식(AAPL, AMZN, GOOGL, MSFT, TSLA)의 일별 주가 데이터를 수집합니다.
2. **CWT 전처리**: 연속 웨이블릿 변환을 적용하여 주가 신호에서 고주파 노이즈를 제거합니다. 임계값 `a_th=24`를 사용합니다.
3. **MA 신호 생성**: 전처리된 신호에 MA3/MA15 크로스오버 전략을 적용하여 매수/매도 신호를 생성합니다.
4. **성능 평가**: ARR, Sharpe Ratio, MDD, 거래 횟수 등의 지표로 Buy-and-Hold 및 기준선 전략과 비교합니다.

**[English]**  
1. **Data Collection**: Collect daily stock price data for 5 US equities (AAPL, AMZN, GOOGL, MSFT, TSLA) via yfinance.
2. **CWT Preprocessing**: Apply Continuous Wavelet Transform to remove high-frequency noise from price signals using threshold `a_th=24`.
3. **MA Signal Generation**: Apply MA3/MA15 crossover strategy to preprocessed signals to generate buy/sell signals.
4. **Performance Evaluation**: Compare against Buy-and-Hold and baseline strategies using ARR, Sharpe Ratio, MDD, and trade frequency.

---

## 라이선스 / License

This project is for academic research purposes.  
본 프로젝트는 학술 연구 목적으로 작성되었습니다.
