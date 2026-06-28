# ============================================================
# MARKET STRESS SIGNAL
# Methods: Financial Turbulence (Kritzman & Li, 2010)
#          Absorption Ratio  (Kritzman, Page & Turkington, 2012)
# Data:    Free via yfinance (no API key needed)
# ============================================================

# ── 0. INSTALL & IMPORT ─────────────────────────────────────
# Run this cell first in Colab:
# !pip install yfinance pandas numpy scikit-learn matplotlib seaborn

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── 1. DEFINE ASSET UNIVERSE ────────────────────────────────
# A diversified multi-asset basket covering:
#   Equities (US large, small, intl, EM)
#   Fixed Income (govts, corp, HY, TIPS)
#   Commodities (gold, oil)
#   Real Assets / Volatility

TICKERS = {
    # US Equities
    "SPY":  "US Large Cap",
    "IWM":  "US Small Cap",
    "QQQ":  "US Tech",
    # International Equities
    "EFA":  "Developed Intl",
    "EEM":  "Emerging Markets",
    # Fixed Income
    "TLT":  "US Long Treasury",
    "IEF":  "US Mid Treasury",
    "LQD":  "Investment Grade Corp",
    "HYG":  "High Yield",
    "TIP":  "TIPS (Inflation Protected)",
    # Commodities & Real Assets
    "GLD":  "Gold",
    "USO":  "Oil",
    "VNQ":  "Real Estate (REITs)",
    # Volatility Proxy
    "VXX":  "VIX Short-Term Futures",
}

START_DATE = "2010-01-01"
END_DATE   = "2024-12-31"

# Known stress episodes to annotate on charts
STRESS_EVENTS = {
    "EU Debt Crisis\n(2011)":    ("2011-07-01", "2011-10-01"),
    "China Selloff\n(2015)":     ("2015-08-01", "2015-09-30"),
    "COVID Crash\n(2020)":       ("2020-02-15", "2020-04-30"),
    "Rate Shock\n(2022)":        ("2022-01-01", "2022-10-31"),
    "SVB Crisis\n(2023)":        ("2023-03-01", "2023-04-30"),
}


# ── 2. DOWNLOAD DATA ────────────────────────────────────────
def download_prices(tickers: dict, start: str, end: str) -> pd.DataFrame:
    """Download adjusted closing prices for all tickers."""
    print("Downloading price data via yfinance …")
    raw = yf.download(list(tickers.keys()), start=start, end=end, auto_adjust=True)
    prices = raw["Close"].dropna(how="all")

    # Forward-fill up to 5 days to handle staggered market holidays
    prices = prices.ffill(limit=5).dropna()
    print(f"  ✓ {prices.shape[1]} assets | {prices.shape[0]} trading days "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


# ── 3. FINANCIAL TURBULENCE ─────────────────────────────────
def financial_turbulence(
    returns: pd.DataFrame,
    lookback: int = 252,        # 1-year estimation window
    min_periods: int = 126,     # 6 months minimum before first estimate
) -> pd.Series:
    """
    Mahalanobis distance of today's return vector from its historical distribution.

    turbulence(t) = (r_t - μ)' Σ⁻¹ (r_t - μ)

    High value  → returns are statistically unusual (stress)
    Low value   → returns within normal historical range (calm)
    """
    turbulence_scores = []
    dates = []

    for i in range(len(returns)):
        if i < min_periods:
            turbulence_scores.append(np.nan)
            dates.append(returns.index[i])
            continue

        # Use a rolling window ending the day BEFORE today (no look-ahead)
        window_start = max(0, i - lookback)
        hist = returns.iloc[window_start:i]

        r_t = returns.iloc[i].values          # today's return vector
        mu  = hist.mean().values              # historical mean
        cov = hist.cov().values               # historical covariance

        try:
            cov_inv = np.linalg.pinv(cov)     # pseudo-inverse handles near-singular matrices
            diff    = r_t - mu
            score   = float(diff @ cov_inv @ diff)
            turbulence_scores.append(max(score, 0))   # floor at 0
        except Exception:
            turbulence_scores.append(np.nan)

        dates.append(returns.index[i])

    series = pd.Series(turbulence_scores, index=dates, name="turbulence_raw")

    # Winsorise extreme outliers at 99.5th percentile so one day doesn't dominate
    cap = series.quantile(0.995)
    series = series.clip(upper=cap)

    return series


# ── 4. ABSORPTION RATIO ─────────────────────────────────────
def absorption_ratio(
    returns: pd.DataFrame,
    lookback: int = 252,
    fraction: float = 0.2,      # fraction of eigenvectors considered "top"
    min_periods: int = 126,
) -> pd.Series:
    """
    Fraction of total variance explained by the top eigenvectors.

    AR = Σ variance(top k PCs) / Σ variance(all PCs)
    where k = ceil(fraction * n_assets)

    High AR → market moves driven by a single common factor (systemic fragility)
    Low AR  → idiosyncratic, diversified (more resilient)
    """
    n_assets = returns.shape[1]
    k = max(1, int(np.ceil(fraction * n_assets)))   # number of "dominant" PCs

    ar_scores = []
    dates     = []

    for i in range(len(returns)):
        if i < min_periods:
            ar_scores.append(np.nan)
            dates.append(returns.index[i])
            continue

        window_start = max(0, i - lookback)
        hist = returns.iloc[window_start:i]

        # Standardise so that assets with different volatility levels
        # contribute equally to the PCA
        scaler = StandardScaler()
        hist_scaled = scaler.fit_transform(hist)

        pca = PCA(n_components=n_assets)
        pca.fit(hist_scaled)

        explained = pca.explained_variance_ratio_
        ar = float(np.sum(explained[:k]))   # variance in top k PCs
        ar_scores.append(ar)
        dates.append(returns.index[i])

    return pd.Series(ar_scores, index=dates, name="absorption_ratio")


# ── 5. COMPOSITE STRESS INDEX ───────────────────────────────
def composite_stress_index(turbulence: pd.Series, ar: pd.Series) -> pd.Series:
    """
    Blend turbulence and absorption ratio into a single [0, 1] stress score.

    Method:
      1. Each series is percentile-ranked over its own history (robust to outliers)
      2. Simple average of the two percentile ranks
      3. Apply a 21-day rolling average to smooth daily noise

    The result can be interpreted as:
      > 0.80  →  high stress / risk-off regime
      0.50–0.80 →  elevated caution
      < 0.50  →  calm / risk-on
    """
    aligned = pd.concat([turbulence, ar], axis=1).dropna()

    # Percentile rank (expanding window so we never look ahead)
    turb_pct = aligned["turbulence_raw"].expanding().rank(pct=True)
    ar_pct   = aligned["absorption_ratio"].expanding().rank(pct=True)

    composite = (turb_pct + ar_pct) / 2

    # 21-day smooth (one calendar month)
    composite_smooth = composite.rolling(21, min_periods=5).mean()
    composite_smooth.name = "stress_index"

    return composite_smooth


# ── 6. REGIME CLASSIFICATION ────────────────────────────────
def classify_regime(stress: pd.Series) -> pd.Series:
    """Label each day as Calm / Elevated / Stress regime."""
    regimes = pd.cut(
        stress,
        bins=[0, 0.50, 0.75, 1.01],
        labels=["Calm", "Elevated", "Stress"],
        right=False,
    )
    return regimes


# ── 7. VISUALISATION ────────────────────────────────────────
PALETTE = {
    "Calm":     "#2ecc71",
    "Elevated": "#f39c12",
    "Stress":   "#e74c3c",
}

def plot_stress_dashboard(
    stress:      pd.Series,
    turbulence:  pd.Series,
    ar:          pd.Series,
    prices:      pd.DataFrame,
    regimes:     pd.Series,
):
    """Four-panel dashboard."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 22), facecolor="#0d1117")
    fig.suptitle(
        "Market Stress Signal\nFinancial Turbulence + Absorption Ratio",
        fontsize=18, color="white", fontweight="bold", y=0.98,
    )

    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # ── Panel 1: Composite Stress Index ─────────────────────
    ax = axes[0]
    ax.set_title("Composite Stress Index  (0 = calm · 1 = extreme stress)", pad=10)

    # Colour-fill by regime
    for regime, colour in PALETTE.items():
        mask = regimes == regime
        ax.fill_between(stress.index, stress, where=mask.reindex(stress.index, fill_value=False),
                        color=colour, alpha=0.35, label=regime)

    ax.plot(stress.index, stress, color="white", linewidth=1.0, alpha=0.9)
    ax.axhline(0.75, color="#e74c3c", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(0.50, color="#f39c12", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Stress Score")
    ax.legend(loc="upper left", facecolor="#161b22", labelcolor="white", framealpha=0.8)

    _annotate_events(ax, STRESS_EVENTS, stress)

    # ── Panel 2: Component Series ────────────────────────────
    ax = axes[1]
    ax.set_title("Components: Turbulence Percentile vs Absorption Ratio Percentile")

    turb_pct = turbulence.expanding().rank(pct=True).rolling(21).mean()
    ar_pct   = ar.expanding().rank(pct=True).rolling(21).mean()

    ax.plot(turb_pct.index, turb_pct, color="#3498db", linewidth=1.2, label="Turbulence %ile")
    ax.plot(ar_pct.index,   ar_pct,   color="#9b59b6", linewidth=1.2, label="Absorption Ratio %ile")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Percentile Rank")
    ax.legend(loc="upper left", facecolor="#161b22", labelcolor="white", framealpha=0.8)

    # ── Panel 3: SPY Drawdown ────────────────────────────────
    ax = axes[2]
    ax.set_title("S&P 500 (SPY) Drawdown from Peak")

    spy = prices["SPY"].reindex(stress.index).ffill()
    drawdown = (spy / spy.expanding().max() - 1) * 100

    ax.fill_between(drawdown.index, drawdown, 0, color="#e74c3c", alpha=0.5)
    ax.plot(drawdown.index, drawdown, color="#e74c3c", linewidth=0.8)
    ax.set_ylabel("Drawdown (%)")

    _annotate_events(ax, STRESS_EVENTS, drawdown)

    # ── Panel 4: Regime Bar Chart ────────────────────────────
    ax = axes[3]
    ax.set_title("Daily Stress Regime")

    regime_num = regimes.map({"Calm": 0.2, "Elevated": 0.6, "Stress": 1.0})
    colour_map = regimes.map(PALETTE)

    ax.bar(regime_num.index, regime_num.values,
           color=colour_map.values, width=1.5, alpha=0.85)
    ax.set_yticks([0.2, 0.6, 1.0])
    ax.set_yticklabels(["Calm", "Elevated", "Stress"])
    ax.set_ylabel("Regime")

    # Date formatting for all panels
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("market_stress_signal.png", dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    plt.show()
    print("  ✓ Chart saved as market_stress_signal.png")


def _annotate_events(ax, events: dict, series: pd.Series):
    """Shade known stress episodes and add labels."""
    ymax = ax.get_ylim()[1]
    for label, (start, end) in events.items():
        try:
            s = pd.Timestamp(start)
            e = pd.Timestamp(end)
            if s < series.index[-1] and e > series.index[0]:
                ax.axvspan(s, e, color="#e74c3c", alpha=0.10)
                mid = s + (e - s) / 2
                ax.text(mid, ymax * 0.92, label, ha="center", va="top",
                        fontsize=7, color="#e74c3c", alpha=0.85)
        except Exception:
            pass


# ── 8. REGIME STATS ─────────────────────────────────────────
def regime_statistics(prices: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    """
    For each stress regime, compute average forward returns
    on SPY over 1-day, 5-day, 21-day windows.
    Shows whether high stress predicts subsequent losses.
    """
    spy = prices["SPY"]
    fwd = {
        "Fwd 1d (%)":  spy.pct_change(1).shift(-1)  * 100,
        "Fwd 5d (%)":  spy.pct_change(5).shift(-5)  * 100,
        "Fwd 21d (%)": spy.pct_change(21).shift(-21) * 100,
    }
    df = pd.DataFrame(fwd).join(regimes.rename("Regime")).dropna()
    stats = df.groupby("Regime")[list(fwd.keys())].agg(["mean", "median", "std"])
    return stats


# ── 9. MAIN ─────────────────────────────────────────────────
def main():
    # Step 1 — Data
    prices  = download_prices(TICKERS, START_DATE, END_DATE)
    returns = compute_returns(prices)

    # Step 2 — Turbulence
    print("\nComputing Financial Turbulence …  (this takes ~1 min in Colab)")
    turb = financial_turbulence(returns, lookback=252, min_periods=126)
    print(f"  ✓ Turbulence computed  |  mean={turb.mean():.2f}  max={turb.max():.2f}")

    # Step 3 — Absorption Ratio
    print("\nComputing Absorption Ratio …  (this takes ~2 min in Colab)")
    ar   = absorption_ratio(returns, lookback=252, fraction=0.20, min_periods=126)
    print(f"  ✓ Absorption Ratio computed  |  mean={ar.mean():.3f}  max={ar.max():.3f}")

    # Step 4 — Composite + Regimes
    stress  = composite_stress_index(turb, ar)
    regimes = classify_regime(stress)

    # Step 5 — Dashboard
    print("\nPlotting dashboard …")
    plot_stress_dashboard(stress, turb, ar, prices, regimes)

    # Step 6 — Regime statistics (the "so what" table)
    print("\n── Forward Return Statistics by Regime ──")
    stats = regime_statistics(prices, regimes)
    print(stats.to_string())

    # Step 7 — Export CSVs
    output = pd.DataFrame({
        "turbulence_raw":   turb,
        "absorption_ratio": ar,
        "stress_index":     stress,
        "regime":           regimes,
    })
    output.to_csv("market_stress_data.csv")
    print("\n  ✓ Data exported to market_stress_data.csv")

    return output, stats


if __name__ == "__main__":
    output, stats = main()
