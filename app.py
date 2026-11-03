"""International cost of capital | Streamlit teaching application.

Run: streamlit run app.py
Market inputs require internet access. Trading Economics public-page extraction
is best-effort, with an explicit manual fallback; no API key is used.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import re

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

BETAS_XLS = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/betas.xls"
BETAS_HTML = "https://pages.stern.nyu.edu/adamodar/New_Home_Page/datafile/Betas.html"
TAX_URL = "https://pages.stern.nyu.edu/adamodar/New_Home_Page/datafile/ctryprem.html"
SPREAD_URL = "https://pages.stern.nyu.edu/adamodar/New_Home_Page/datafile/ratings.html"
COUNTRIES = {
    "United States": ("USD", "united-states"),
    "Australia": ("AUD", "australia"), "Brazil": ("BRL", "brazil"),
    "Canada": ("CAD", "canada"), "China": ("CNY", "china"),
    "France": ("EUR", "france"), "Germany": ("EUR", "germany"),
    "India": ("INR", "india"), "Italy": ("EUR", "italy"),
    "Japan": ("JPY", "japan"), "Mexico": ("MXN", "mexico"),
    "South Korea": ("KRW", "south-korea"), "Switzerland": ("CHF", "switzerland"),
    "United Kingdom": ("GBP", "united-kingdom"),
}

st.set_page_config(page_title="International Cost of Capital", page_icon="🌐", layout="wide")
st.title("🌐 International Cost of Capital")
st.caption("A transparent, currency-consistent cost-of-capital calculator • rates and spreads are annual percentages")


with st.expander("Model equations and assumptions", expanded=True):
    st.latex(r"\beta_L=\beta_U\left[1+(1-T)\frac{D}{E}\right],\quad \frac{D}{E}=\frac{w_D}{1-w_D}")
    st.latex(r"k_{E,c}=r_{10Y,c}+\beta_L\,ERP")
    st.latex(r"s_{FC}^{auto}=s_{USD}\frac{1+r_{10Y,FC}}{1+r_{10Y,USD}}\quad\text{(rates in decimals)}")
    st.latex(r"k_{D,c}=r_{10Y,c}+s_c,\qquad WACC_c=(1-w_D)k_{E,c}+w_Dk_{D,c}(1-T)")
    st.markdown("**Interpretation:** All rates use the same project beta, ERP, tax rate and debt/equity weights. We proxy each currency's base rate with its 10-year sovereign yield. The automatic spread rule is an *explicit modeling convention* that preserves the USD debt premium after a gross-rate currency translation; it is not an observed local-currency BBB credit spread. Local sovereign risk, currency hedging costs, country risk premia, and cross-currency basis are **not** separately modeled. A different formula supplied in class should supersede this illustrative convention.")


def get_page(url: str) -> str:
    response = requests.get(url, timeout=18, headers={"User-Agent": "Mozilla/5.0 (education; financial-data exercise)"})
    response.raise_for_status()
    return response.text


def parse_rate(value) -> float:
    """Return percentage point value (e.g. '25%' -> 25.0)."""
    v = str(value).replace(",", "").replace("%", "").strip()
    if v.lower() in {"", "nan", "na", "n/a", "none"}:
        raise ValueError("Missing rate")
    return float(v)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r"\s+", " ", str(c)).strip().lower() for c in out.columns]
    return out


def beta_from_html(html: str) -> pd.DataFrame:
    for raw in pd.read_html(BytesIO(html.encode("utf-8"))):
        df = clean_columns(raw)
        inds = next((c for c in df if "industry name" in c), None)
        betas = next((c for c in df if c == "unlevered beta"), None)
        if inds and betas:
            table = pd.DataFrame({"Industry": df[inds].astype(str).str.strip(),
                                  "Unlevered beta": pd.to_numeric(df[betas], errors="coerce")})
            return table.dropna().query("`Unlevered beta` >= 0").drop_duplicates("Industry").reset_index(drop=True)
    raise ValueError("Couldn't locate the industry/unlevered-beta table")


def load_betas() -> tuple[pd.DataFrame, str]:
    try:
        r = requests.get(BETAS_XLS, timeout=18, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        xls = pd.ExcelFile(BytesIO(r.content))
        for sheet in xls.sheet_names:
            for row in range(0, 12):
                df = pd.read_excel(xls, sheet_name=sheet, header=row)
                df.columns = [re.sub(r"\s+", " ", str(c)).strip().lower() for c in df.columns]
                name = next((c for c in df if c == "industry name"), None)
                beta = next((c for c in df if c == "unlevered beta"), None)
                if name and beta:
                    result = pd.DataFrame({"Industry": df[name].astype(str).str.strip(),
                                           "Unlevered beta": pd.to_numeric(df[beta], errors="coerce")})
                    result = result.dropna().query("`Unlevered beta` >= 0").drop_duplicates("Industry")
                    if len(result) > 20:
                        return result.reset_index(drop=True), BETAS_XLS
    except Exception:
        pass
    return beta_from_html(get_page(BETAS_HTML)), BETAS_HTML


def parse_country_taxes(html: str) -> dict[str, float]:
    """Read the country/tax column, tolerating nonstandard Damodaran HTML.

    First match explicit header columns. If the HTML uses unusual headers,
    parse country rows by their published ordering: country, Moody's rating,
    adjusted default spread, country risk premium, ERP, corporate tax rate.
    Fail rather than silently returning an unvalidated column.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"], recursive=False)]
        if cells:
            rows.append(cells)

    def norm(value):
        return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()

    # First strategy: explicitly identify the table column names.
    for i, row in enumerate(rows):
        normalized = [norm(c) for c in row]
        country_idx = next((j for j, c in enumerate(normalized) if c == "country"), None)
        tax_idx = next((j for j, c in enumerate(normalized)
                        if "corporate" in c and "tax" in c and "rate" in c), None)
        if country_idx is None or tax_idx is None:
            continue
        result = {}
        for candidate in rows[i+1:]:
            if len(candidate) <= max(country_idx, tax_idx):
                continue
            country_name = candidate[country_idx].strip()
            try:
                value = parse_rate(candidate[tax_idx])
            except (TypeError, ValueError):
                continue
            if country_name and 0 <= value <= 100:
                result[country_name] = value
        if len(result) >= 10:
            return result

    # Second strategy: rows may render as a single combined text element.
    # Moody's ratings and the FOUR consecutive percentage columns give a
    # structural validation: the fourth is the corporate tax rate.
    rating = r"(?:Aaa|Aa[123]|A[123]|Baa[123]|Ba[123]|B[123]|Caa[123]|Ca|C|NR)"
    pct = r"(-?\d+(?:\.\d+)?)\s*%"
    row_re = re.compile(
        rf"^(.+?)\s+({rating})\s+{pct}\s+{pct}\s+{pct}\s+{pct}(?:\s|$)",
        re.I,
    )
    result = {}
    for row in rows:
        joined = " ".join(row)
        joined = re.sub(r"\s+", " ", joined).strip()
        match = row_re.match(joined)
        if match:
            country_name = match.group(1).strip()
            value = float(match.group(6))
            if country_name and 0 <= value <= 100:
                result[country_name] = value
    if len(result) >= 10:
        return result

    sample = [" | ".join(row[:8])[:160] for row in rows[:4]]
    raise ValueError(
        f"Tax-rate table was not identified ({len(rows)} HTML rows; "
        f"{len(result)} rating-pattern matches). First rows: {sample}"
    )


def load_tax() -> dict[str, float]:
    return parse_country_taxes(get_page(TAX_URL))


def load_bbb_spread() -> float:
    """Use large non-financial-firm BBB (Baa2/BBB) default spread, percent points."""
    soup = BeautifulSoup(get_page(SPREAD_URL), "html.parser")
    for row in soup.select("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"], recursive=False)]
        if any(re.search(r"\bBaa2\s*/\s*BBB\b", c, flags=re.I) for c in cells):
            # In Damodaran table, first spread to the right of the FIRST BBB rating belongs to non-financials.
            idx = next(i for i, c in enumerate(cells) if re.search(r"\bBaa2\s*/\s*BBB\b", c, re.I))
            for c in cells[idx+1:]:
                m = re.search(r"(-?\d+(?:\.\d+)?)\s*%", c)
                if m:
                    return float(m.group(1))
    raise ValueError("BBB spread was not found in Damodaran's ratings table")


def parse_te_public_page(html: str, country: str) -> tuple[float, str]:
    """Extract an observed 10Y quote from a public country yield page.

    Rejects pages whose header/quote cannot be verified. This is best-effort,
    NOT an official API or guaranteed to work on Streamlit Cloud.
    """
    soup = BeautifulSoup(html, "html.parser")
    # Strip scripts so embedded forecasts and JSON cannot masquerade as a quote.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    heading = soup.find("h1")
    heading_text = heading.get_text(" ", strip=True) if heading else ""
    if "10" not in heading_text or not re.search(r"bond|treasury|yield", heading_text, re.I):
        raise ValueError(f"10-year bond page heading not found for {country}")
    text = soup.get_text(" ", strip=True)
    # 'Actual' is the main current quote; take the FIRST one immediately after the heading.
    start = text.find(heading_text)
    relevant = text[start : start + 450] if start >= 0 else ""
    m = re.search(r"\bActual\s+(-?\d{1,2}(?:\.\d{1,5})?)\b\s+Daily\s+Change\b", relevant, re.I)
    if not m:
        raise ValueError("10-year Actual yield quote not found in public HTML")
    yield_pct = float(m.group(1))
    if not -20 <= yield_pct <= 40:
        raise ValueError("Public yield outside conservative validation range")
    # This is an observation date only if stated explicitly on the page.
    # Prefer the observation date tied to the 10Y yield quote, rather than a date
    # from an unrelated news story. If unavailable, leave the date unverified.
    stats = re.search(r"(?:10.year government bond yield|10Y bond yield)[^.]{0,150}?\bon\s+([A-Z][a-z]+\s+\d{1,2},?\s+20\d{2})", text, re.I)
    updated = re.search(r"(?:data, forecasts, historical chart\s+-\s+was last updated on)\s+([A-Z][a-z]+\s+\d{1,2}(?:\s+of)?\s+20\d{2})", text, re.I)
    observation = (stats.group(1) if stats else updated.group(1) if updated else "not verified on source page")
    return yield_pct, observation


@st.cache_data(ttl=3600, show_spinner=False)
def load_te_public_yield(slug: str, country: str) -> tuple[float, str]:
    url = f"https://tradingeconomics.com/{slug}/government-bond-yield"
    return parse_te_public_page(get_page(url), country)


def calculate(beta_u: float, debt_weight: float, tax: float, erp: float,
              rf_us: float, rf_foreign: float, spread_us: float,
              spread_fc_override: float | None = None) -> dict[str, float]:
    """Arguments are percentages except beta and debt weight (0 to 1).

    Foreign spread uses multiplicative currency conversion of the *total* debt rate.
    This is a modeling convention; sovereign risk / CIP basis are not modeled.
    """
    assert 0 <= debt_weight < 1 and 0 <= tax <= 100
    d_e = debt_weight / (1 - debt_weight)
    beta_l = beta_u * (1 + (1 - tax / 100) * d_e)
    cost_e_us = rf_us + beta_l * erp
    cost_e_fc = rf_foreign + beta_l * erp
    cost_d_us = rf_us + spread_us
    # Same USD credit component translated with relative gross base yields.
    spread_fc_auto = spread_us * (100 + rf_foreign) / (100 + rf_us)
    spread_fc = spread_fc_auto if spread_fc_override is None else spread_fc_override
    cost_d_fc = rf_foreign + spread_fc
    wacc_us = (1 - debt_weight) * cost_e_us + debt_weight * cost_d_us * (1 - tax / 100)
    wacc_fc = (1 - debt_weight) * cost_e_fc + debt_weight * cost_d_fc * (1 - tax / 100)
    return dict(beta_l=beta_l, spread_fc_auto=spread_fc_auto, spread_fc=spread_fc,
                cost_e_us=cost_e_us, cost_e_fc=cost_e_fc,
                cost_d_us=cost_d_us, cost_d_fc=cost_d_fc,
                wacc_us=wacc_us, wacc_fc=wacc_fc)


@st.cache_data(ttl=86400, show_spinner=False)
def get_damodaran():
    results, errors = {}, {}
    for key, fn in (("betas", load_betas), ("tax", load_tax), ("spread", load_bbb_spread)):
        try:
            results[key] = fn()
        except Exception as exc:
            errors[key] = str(exc)
    return results, errors


st.sidebar.header("Parameters & overrides")
refresh = st.sidebar.button("↻ Refresh external data", help="Clears cached Damodaran inputs and reloads the Trading Economics snapshot")
if refresh:
    get_damodaran.clear()
    load_te_public_yield.clear()
    st.rerun()
with st.spinner("Reading published data..."):
    sources, errors = get_damodaran()

if errors:
    st.sidebar.warning("Some Damodaran inputs are unavailable; use the manual inputs below. " + "; ".join(f"{k}: {v}" for k, v in errors.items()))

all_betas = sources.get("betas", (pd.DataFrame(), ""))[0]
industries = all_betas["Industry"].tolist() if len(all_betas) else ["Software (System & Application)"]
preferred = "Software (System & Application)"
st.sidebar.markdown("### Project selection")
industry = st.sidebar.selectbox("U.S. industry", industries, index=industries.index(preferred) if preferred in industries else 0)
country = st.sidebar.selectbox("Target country (U.S. is fixed baseline)", list(COUNTRIES), index=0)
currency, slug = COUNTRIES[country]

industry_beta = float(all_betas.set_index("Industry").loc[industry, "Unlevered beta"]) if len(all_betas) and industry in all_betas["Industry"].values else None
# Like published tax and BBB spread, the industry beta defaults to the
# Damodaran value. Manual editing is explicitly opt-in (or a data fallback).
st.sidebar.markdown("### Unlevered beta")
if industry_beta is not None:
    st.sidebar.markdown(f"**{industry_beta:.3f}** · Damodaran industry beta")
    beta_override = st.sidebar.checkbox("Override unlevered beta", value=False, key="override_beta")
else:
    st.sidebar.warning("Published unlevered beta unavailable; enter a value below.")
    beta_override = True
if beta_override:
    beta_u = st.sidebar.number_input(
        "Project unlevered beta", min_value=0.0, max_value=10.0,
        value=float(industry_beta) if industry_beta is not None else 1.0,
        step=0.05, format="%.3f", key=f"manual_beta_{industry}",
        help="Use a manual beta only for a deliberate override or when Damodaran data is unavailable."
    )
else:
    beta_u = float(industry_beta)
st.sidebar.markdown("### Equity risk premium")
erp = st.sidebar.number_input("ERP (%)", 0.0, 40.0, 6.5, 0.25)
st.sidebar.markdown("### Financing ratio")
debt_pct = st.sidebar.slider("Debt share of total capital (%)", 0, 95, 30)
st.sidebar.caption(f"Debt: {debt_pct}% · Equity: {100-debt_pct}% · D/E: {debt_pct / (100-debt_pct):.3f}")

# Load both tax rates from the same country table. The United States is a
# fixed baseline, not a selectable target-market replacement.
tax_table = sources.get("tax", {})
published_us_tax = tax_table.get("United States")
published_target_tax = tax_table.get("Korea" if country == "South Korea" else country)
published_spread = sources.get("spread")

st.sidebar.markdown("### Project tax rate")
st.sidebar.caption("One project tax rate applies to both currencies.")
tax_jurisdiction = st.sidebar.radio(
    "Project taxable in", ["Target country", "United States"],
    index=0, horizontal=False,
    help="This selects a single tax rate used in both USD and foreign-currency calculations."
)
tax_country = country if tax_jurisdiction == "Target country" else "United States"
published_tax = published_target_tax if tax_jurisdiction == "Target country" else published_us_tax

# Published Damodaran values are used automatically. Overrides only appear
# when the user explicitly requests one, or when downloading fails.
if published_tax is not None:
    st.sidebar.markdown(f"**{published_tax:.2f}%** · {tax_country} (Damodaran)")
    tax_override = st.sidebar.checkbox("Override corporate tax rate", value=False, key="override_tax")
else:
    st.sidebar.warning(f"No published tax rate available for {tax_country}; enter a value below.")
    tax_override = True
if tax_override:
    tax = st.sidebar.number_input("Project corporate tax rate (%)", 0.0, 100.0,
                                  value=float(published_tax) if published_tax is not None else 25.0,
                                  step=0.5, key=f"manual_tax_{tax_country}")
else:
    tax = float(published_tax)

st.sidebar.markdown("### USD BBB spread")
if published_spread is not None:
    st.sidebar.markdown(f"**{published_spread:.3f}%** · Damodaran")
    spread_override = st.sidebar.checkbox("Override USD BBB spread", value=False, key="override_us_spread")
else:
    st.sidebar.warning("Published BBB USD spread unavailable; enter a value below.")
    spread_override = True
if spread_override:
    spread_us = st.sidebar.number_input("BBB USD borrowing spread (%)", 0.0, 30.0,
                                        value=float(published_spread) if published_spread is not None else 1.11,
                                        step=0.05, key="manual_us_spread")
else:
    spread_us = float(published_spread)

# Yield inputs: concise sidebar display. Source details remain in the Input Audit.
st.sidebar.markdown("### 10-year government yields")
use_public = True  # Public-page retrieval is always attempted; no API key.

def resolve_yield(country_name: str, country_slug: str):
    try:
        val, date = load_te_public_yield(country_slug, country_name)
        return val, date, "Trading Economics public page", None
    except Exception as exc:
        return None, None, None, str(exc)

us_value, us_source_date, us_source, us_error = resolve_yield("United States", "united-states")
if country == "United States":
    fc_value, fc_source_date, fc_source, fc_error = us_value, us_source_date, us_source, us_error
else:
    fc_value, fc_source_date, fc_source, fc_error = resolve_yield(country, slug)

# Show the retrieved value and its observation date before optional editing.
def yield_control(label: str, quote: float | None, source_date: str | None,
                  override_key: str, value_key: str, date_key: str,
                  fallback: float) -> tuple[float, str, bool]:
    st.sidebar.markdown(f"**{label}**")
    if quote is not None:
        st.sidebar.caption(f"Trading Economics: **{quote:.4f}%** · Date: {source_date or 'unverified'}")
        manual = st.sidebar.checkbox(f"Override {label} yield", value=False, key=override_key)
    else:
        st.sidebar.caption("Trading Economics: **Unavailable** · Date: unavailable")
        st.sidebar.warning("Automatic retrieval failed; manual entry required.")
        manual = True
    if manual:
        val = st.sidebar.number_input(
            f"{label} yield (%)", min_value=-20.0, max_value=100.0,
            value=float(quote if quote is not None else fallback), step=0.1,
            key=value_key,
        )
        date = st.sidebar.text_input(f"{label} observation date", value="not provided", key=date_key)
    else:
        val, date = float(quote), source_date or "unverified"
    return float(val), date, manual

rf_us, us_date, us_manual = yield_control(
    "United States", us_value, us_source_date, "us_manual", "us_yield", "us_date", 4.0
)
rf_foreign, fc_date, fc_manual = yield_control(
    country, fc_value, fc_source_date, f"fc_manual_{country}",
    f"fc_yield_{country}", f"fc_date_{country}",
    rf_us if country == "United States" else 3.0,
)
# Technical failure reasons belong in a collapsible diagnostic, not the main form.
if us_error or fc_error:
    with st.sidebar.expander("Yield retrieval details"):
        for name, err in (("United States", us_error), (country, fc_error)):
            if err:
                st.caption(f"{name}: {err}")

if country == "United States" and abs(rf_us - rf_foreign) > 1e-8:
    st.sidebar.warning("Both columns use USD; set both government yields equal for currency consistency.")

debt_weight = debt_pct / 100
base = calculate(beta_u, debt_weight, tax, erp, rf_us, rf_foreign, spread_us)
st.sidebar.markdown("### Target-currency BBB spread")
st.sidebar.caption(f"Automatic BBB {currency} spread: **{base['spread_fc_auto']:.3f}%**")
spread_override_on = st.sidebar.checkbox("Override converted foreign-currency spread", value=False)
if spread_override_on:
    spread_fc_override = st.sidebar.number_input(
        f"Manual BBB spread in {currency} (%)", min_value=0.0, max_value=30.0,
        value=float(base["spread_fc_auto"]), step=0.05)
else:
    spread_fc_override = None
values = calculate(beta_u, debt_weight, tax, erp, rf_us, rf_foreign, spread_us,
                   spread_fc_override)
st.sidebar.caption(f"Used: **{values['spread_fc']:.3f}%** ({'manual override' if spread_override_on else 'automatic conversion'})")

# Present the published inputs for BOTH countries in one aligned table.
# The tax observations are reference data; the model uses ONE project tax rate
# in both WACC columns to isolate currency denomination, per assignment.
st.subheader("Input audit: United States (fixed) vs. target")
st.caption("Published tax rates are shown for BOTH jurisdictions, but the same selected project tax rate enters both calculations. U.S. industry beta and the ERP are also common to both currency calculations.")
audit_rows = [
    {"Input": "Currency", "United States (fixed)": "USD", f"Target: {country}": currency},
    {"Input": "10Y sovereign yield (%)", "United States (fixed)": f"{rf_us:.4f}%", f"Target: {country}": f"{rf_foreign:.4f}%"},
    {"Input": "Yield source", "United States (fixed)": "manual" if us_manual else (us_source or "unavailable"), f"Target: {country}": "manual" if fc_manual else (fc_source or "unavailable")},
    {"Input": "Yield observation date", "United States (fixed)": us_date or "unverified", f"Target: {country}": fc_date or "unverified"},
    {"Input": "Published corporate tax (%)", "United States (fixed)": f"{published_us_tax:.2f}%" if published_us_tax is not None else "unavailable", f"Target: {country}": f"{published_target_tax:.2f}%" if published_target_tax is not None else "unavailable"},
    {"Input": "Tax actually used in BOTH WACCs (%)", "United States (fixed)": f"{tax:.2f}% ({tax_country})", f"Target: {country}": f"{tax:.2f}% ({tax_country})"},
    {"Input": "Unlevered industry beta (U.S. industry dataset)", "United States (fixed)": f"{beta_u:.3f}", f"Target: {country}": f"{beta_u:.3f} (shared)"},
    {"Input": "Levered project beta", "United States (fixed)": f"{values['beta_l']:.3f}", f"Target: {country}": f"{values['beta_l']:.3f} (shared)"},
    {"Input": "Equity risk premium (%)", "United States (fixed)": f"{erp:.2f}%", f"Target: {country}": f"{erp:.2f}% (shared)"},
    {"Input": "BBB borrowing spread (%)", "United States (fixed)": f"{spread_us:.3f}% (Damodaran / override)", f"Target: {country}": f"{values['spread_fc']:.3f}% ({'override' if spread_override_on else 'converted'})"},
    {"Input": "Debt weight (%)", "United States (fixed)": f"{debt_pct:.0f}%", f"Target: {country}": f"{debt_pct:.0f}% (shared)"},
    {"Input": "Equity weight (%)", "United States (fixed)": f"{100-debt_pct:.0f}%", f"Target: {country}": f"{100-debt_pct:.0f}% (shared)"},
]
st.dataframe(pd.DataFrame(audit_rows).set_index("Input"), width="stretch")
st.caption("Damodaran source links and release context are listed at the bottom. Government yields are source-dated individually; 'unverified' is not a confirmed market observation date.")

st.subheader("Side-by-side valuation inputs")
left, right = st.columns(2)
for pane, label, suffix in ((left, "Original: United States (USD)", "us"), (right, f"Target: {country} ({currency})", "fc")):
    with pane:
        st.markdown(f"### {label}")
        st.metric("Cost of equity", f"{values[f'cost_e_{suffix}']:.2f}%")
        st.metric("Pre-tax cost of debt", f"{values[f'cost_d_{suffix}']:.2f}%")
        st.metric("WACC", f"{values[f'wacc_{suffix}']:.2f}%")
st.caption(f"Levered beta (shared): {values['beta_l']:.3f} | Debt weight (shared): {debt_pct:.0f}% | Equity weight (shared): {100-debt_pct:.0f}% | Project tax in BOTH calculations: {tax:.2f}% ({tax_country})")


st.subheader("All-industry comparison")
if all_betas.empty:
    st.error("Industry comparison requires the Damodaran industry-beta data. Please refresh when connected to the internet.")
else:
    rows = []
    for row in all_betas.itertuples(index=False):
        b = float(row[1])
        z = calculate(b, debt_weight, tax, erp, rf_us, rf_foreign, spread_us,
                      spread_fc_override if spread_override_on else None)
        rows.append({"Industry": row[0], "Unlevered beta": b, "Levered beta": z["beta_l"],
                     "USD cost of equity (%)": z["cost_e_us"], "USD cost of debt (%)": z["cost_d_us"],
                     "USD WACC (%)": z["wacc_us"], f"{currency} cost of equity (%)": z["cost_e_fc"],
                     f"{currency} cost of debt (%)": z["cost_d_fc"], f"{currency} WACC (%)": z["wacc_fc"]})
    industry_df = pd.DataFrame(rows).sort_values("USD WACC (%)")
    st.caption("Only industry beta varies. Country, risk-free yields, tax rate, ERP, financing weights and spreads are held constant.")
    st.dataframe(industry_df, width="stretch", hide_index=True)
    st.download_button("Download all-industry results (CSV)", industry_df.to_csv(index=False).encode(),
                       file_name="international_cost_of_capital_industries.csv", mime="text/csv")

selected_row = pd.DataFrame([{
    "industry": industry, "original_country": "United States", "country": country, "currency": currency,
    "us_published_tax_pct": published_us_tax, "target_published_tax_pct": published_target_tax,
    "project_tax_jurisdiction": tax_country,
    "unlevered_beta": beta_u, "levered_beta": values["beta_l"], "erp_pct": erp,
    "debt_weight": debt_weight, "tax_pct": tax, "us_10y_yield_pct": rf_us,
    "country_10y_yield_pct": rf_foreign, "us_bbb_spread_pct": spread_us,
    "fc_bbb_spread_pct": values["spread_fc"], "fc_bbb_spread_override": spread_override_on,
    "us_cost_equity_pct": values["cost_e_us"], "us_cost_debt_pct": values["cost_d_us"],
    "us_yield_source": "manual" if us_manual else us_source, "us_observation_date": us_date,
    "tax_source": "manual override" if tax_override else f"Damodaran ({tax_country})",
    "us_bbb_spread_source": "manual override" if spread_override else "Damodaran",
    "fc_yield_source": "manual" if fc_manual else fc_source, "fc_observation_date": fc_date,
    "us_wacc_pct": values["wacc_us"], "fc_cost_equity_pct": values["cost_e_fc"],
    "fc_cost_debt_pct": values["cost_d_fc"], "fc_wacc_pct": values["wacc_fc"],
}])
st.download_button("Download current project result (CSV)", selected_row.to_csv(index=False).encode(),
                   file_name="international_cost_of_capital_project.csv", mime="text/csv")

st.subheader("Data sources and provenance")
st.markdown(f"- [Damodaran U.S. industry betas]({sources.get('betas', (None, BETAS_XLS))[1] if 'betas' in sources else BETAS_XLS}) — published January 2026 (verify release on source)\n- [Damodaran corporate tax rates]({TAX_URL}) — country risk-premia table, January 2026\n- [Damodaran BBB borrowing spread]({SPREAD_URL}) — large non-financial company row, January 2026\n- [Trading Economics 10-year government yields](https://tradingeconomics.com/united-states/government-bond-yield) — source and date shown above for public HTML; manual input otherwise")
st.caption(f"App checked at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} • This is not a live quote timestamp.")
if "betas" not in sources or tax_override or spread_override or us_manual or fc_manual:
    st.info("At least one figure is manually specified or an example fallback. Confirm source links and observation dates before submitting financial conclusions.")
