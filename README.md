# PlainCents

Personal finance and investment analytics: ML-driven spending intelligence and reporting from your bank CSV exports.

## Goals

- **Ingest** raw bank CSV exports (TD, RBC, Scotiabank) and normalize dates/columns.
- **Categorize** transactions with K-Means clustering (8 categories, label mapping, held-out evaluation).
- **Forecast** 3-month spending per category with Random Forest and walk-forward validation.
- **Persist** everything in a local SQLite data warehouse (6 tables), with price caching for portfolio P&L.
- **Report** via an automated Matplotlib PDF and an interactive PowerBI dashboard.

Python pipeline + SQLite + PowerBI + PDF. (Local Python ML pipeline ,o web hosting.)

## Tech stack

Python, Pandas, NumPy, scikit-learn, SQLite, yfinance, Matplotlib, PowerBI.

## Getting started

_(Setup and run instructions will be added as the pipeline is built. For now: install dependencies with `pip install -r requirements.txt`.)_

## Data privacy

Never commit real bank data. `data/raw/` is gitignored; only synthetic data is included in the repo.
