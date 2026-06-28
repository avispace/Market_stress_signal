# Market Stress Signal

A quantitative market-stress indicator built from two systemic-risk methods:

| Method | What it measures |
|---|---|
| **Financial Turbulence** (Kritzman & Li, 2010) | How statistically *unusual* today's asset returns are vs history |
| **Absorption Ratio** (Kritzman, Page & Turkington, 2012) | How *unified* market moves are — a proxy for systemic fragility |

The two signals are blended into a single **Composite Stress Index** scored 0–1, and each day is classified into a **Calm / Elevated / Stress** regime.

---

## Output

![Market Stress Dashboard](market_stress_signal.png)

---

## Asset Universe

14 ETFs spanning US equities, international equities, fixed income, commodities, and real assets.  
Data is free via `yfinance` — no API key required.

| Ticker | Asset Class |
|---|---|
| SPY, IWM, QQQ | US Equities |
| EFA, EEM | International Equities |
| TLT, IEF, LQD, HYG, TIP | Fixed Income |
| GLD, USO | Commodities |
| VNQ | Real Estate |
| VXX | Volatility |

---

## Quickstart (Google Colab)

```python
# 1. Install dependencies
!pip install yfinance pandas numpy scikit-learn matplotlib seaborn

# 2. Upload market_stress_signal.py to your Colab session
# 3. Run
from market_stress_signal import main
output, stats = main()
```

Or just open the notebook directly:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## How It Works

### Financial Turbulence
Computes the Mahalanobis distance of today's return vector from its historical distribution:

```
turbulence(t) = (r_t − μ)ᵀ Σ⁻¹ (r_t − μ)
```

A high score means asset returns are behaving in ways that are statistically rare — either abnormally large moves, abnormal correlations between assets, or both.

### Absorption Ratio
Uses PCA on a rolling window of returns. The ratio of variance explained by the top 20% of eigenvectors measures how much market movement is driven by a single common factor:

```
AR = Σ variance(top k PCs) / Σ variance(all PCs)
```

A high ratio means the market is "unified" — diversification breaks down and shocks propagate system-wide.

### Composite Index
1. Each series is **percentile-ranked** over its expanding history (no look-ahead bias)
2. The two percentile ranks are **averaged**
3. A **21-day rolling mean** smooths daily noise

---

## Regime Definitions

| Regime | Stress Score | Interpretation |
|---|---|---|
| 🟢 Calm | < 0.50 | Risk-on, normal market conditions |
| 🟡 Elevated | 0.50 – 0.75 | Heightened caution warranted |
| 🔴 Stress | > 0.75 | Risk-off, systemic fragility elevated |

---

## References

- Kritzman, M. & Li, Y. (2010). *Skulls, Financial Turbulence, and Risk Management*. Financial Analysts Journal.
- Kritzman, M., Page, S. & Turkington, D. (2012). *Regime Shifts: Implications for Dynamic Strategies*. Financial Analysts Journal.

---

## Author

Avipsa Acharya · [LinkedIn](https://www.linkedin.com/in/avipsa-acharya)
