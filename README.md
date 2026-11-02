# International Cost of Capital — Streamlit App

Compare the cost of equity, cost of debt and WACC for the **same project** when expressed in USD versus a chosen country's currency. The app lets students alter assumptions, inspect data provenance, compare industry betas, and export results.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Or push `app.py` and `requirements.txt` to a GitHub repository and deploy with Streamlit Community Cloud. Python 3.11 or 3.12 recommended.

## Market data

- Damodaran industry betas: XLS table; if blocked, fallback to the published HTML industry table.
- Damodaran country corporate tax rates: the `Corporate Tax Rate` column of the country premium table.
- Damodaran BBB spread: `Baa2/BBB` row for **large non-financial companies**.
- Trading Economics 10-year government yields: optional authenticated API, or dated manual inputs from the source web pages.

To enable automatically retrieved bond yields, set an environment variable:

```bash
export TRADING_ECONOMICS_API_KEY='your-key-here'
streamlit run app.py
```

Or set `TRADING_ECONOMICS_API_KEY` in Streamlit's app secrets (never commit your secret to Git).

**No API key?** Both 10-year yields can be entered manually; their observation dates can be recorded. The initial 4.00% and 3.00% form values are deliberately illustrative, **not current quotes**. A failed Damodaran request also triggers explicit manual/example fallback labeling. Use **Refresh external data** to clear the Damodaran cache. Note that automated TE retrieval fetches a fresh snapshot on each app rerun.

## Controls and results

- Choose an industry (defaults to Software (System & Application) when present) and a country (defaults to U.S.).
- Change beta, 6.5% ERP default, debt weight (30% default), country tax rate and USD BBB spread.
- See the foreign spread *automatically calculated* with an independent optional manual override; turn the override OFF to restore automatic calculations.
- Compare cost of equity, debt and WACC in USD and selected currency, plus an all-industry table and downloadable CSV files.
- Expand the equations panel for assumptions and diagnostics.

## Scope and limitation

Countries in the selector are a curated set of frequently used currencies; expansion requires adding entries to the `COUNTRIES` mapping. A country's government yield is a proxy for a risk-free rate, not necessarily a true default-free rate. Foreign currency comparisons require a consistent set of assumptions and should **not** be interpreted as proof that local funding is cheaper. The auto spread translation is **illustrative**; the class-prescribed formula, if different, takes precedence.

## Verification

Run:

```bash
python test_calculations.py
```

See `verification_note.md` for a complete independently worked example and all three assignment test cases.
