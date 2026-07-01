# SEC Data Analysis — Can SEC Form 3 Filings Predict Short-Term Stock Movements?

This project investigates whether corporate insider transactions (filed via SEC Form 3) contain
a statistically learnable signal that predicts whether a stock price rises within 5 days of the filing.
The full pipeline covers data cleaning, exploratory analysis, clustering, and supervised classification
using traditional ML models and a neural network.

---

## Motivation

Corporate Insiders such as executives, directors, and major shareholders are required to publicly disclose
their trades within two business days via SEC Form 3. The hypothesis: insiders act on non-public
knowledge, and their trading patterns (size, frequency, role) may carry a predictive signal
for short-term price movements.

---

## Project Structure
<img width="1731" height="1002" alt="ulm drawio(2)" src="https://github.com/user-attachments/assets/9f1b5748-f447-46ae-8903-84c284f240ca" />



## Key Findings

- **AUC consistently hovers around 0.54–0.7** where xgboost performs best out of linear regression, random forest and a simple neuronal net
- **Logistic Regression performs comparably to tree-based models**, suggesting the signal
  is weak and approximately linear if it exists at all.
- **Cluster membership** derived from transaction characteristics proved to be among the
  more informative features, hinting that transaction archetype matters more than individual
  transaction size alone.
- **Hyperparameter tuning did not substantially improve AUC**, reinforcing that the bottleneck
  is feature informativeness rather than model complexity.
- The most predictive features across models were transformed versions of **trade size**
  (`boxcox_amounts_shares`), **price per share** (`boxcox_amounts.pricePerShare`), and
  **recent trading activity** (`scaled_trades_14d`).

> **Interpretation:** SEC Form 3 data in isolation appears to carry only a very weak
> short-term price signal. This is consistent with semi-strong market efficiency,
> publicly available insider filings are quickly priced in. A stronger signal might
> emerge by combining these features with sentiment data, sector context, or longer
> prediction horizons.

---

## Tech Stack

| Area | Libraries |
|---|---|
| Data processing | `pandas`, `numpy`, `scipy` |
| ML models | `scikit-learn`, `xgboost` |
| Neural network | `tensorflow` / `keras` |
| Hyperparameter tuning | `optuna` |
| Visualization | `matplotlib`, `seaborn` |

---

## Limitations & Next Steps

- The 5-day binary target is a simplistic formulation; a different approach may yield more insight
- No sentiment, news, or macro features were incorporated

