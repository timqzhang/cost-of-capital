"""
International Cost of Capital App
---------------------------------
Compares a project's cost of equity, cost of debt and WACC in USD and in the
currency of a selected country.

Data sources
  * Unlevered industry betas  : Damodaran, betas.xls (U.S.)
  * Corporate tax rates       : Damodaran, ctryprem.html
  * USD BBB spread            : Damodaran, ratings.html (Baa2/BBB, large non-financial firms)
  * 10-year government yields : Trading Economics, /bonds

Model
  D/E      = wD / (1 - wD)
  beta_L   = beta_U * [1 + (1 - T) * D/E]
  kE,c     = r10Y,c + beta_L * ERP
  sFC_auto = sUSD * (1 + r10Y,FC) / (1 + r10Y,USD)      (rates in decimals)
  kD,c     = r10Y,c + s_c
  WACC_c   = (1 - wD) * kE,c + wD * kD,c * (1 - T)
"""

from __future__ import annotations

import datetime as dt
import io
import re
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="International Cost of Capital",
    page_icon="🌍",
    layout="wide",
)

BETA_URLS = [
    "https://pages.stern.nyu.edu/~adamodar/pc/datasets/betas.xls",
]
TAX_URLS = [
    "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html",
    "https://pages.stern.nyu.edu/adamodar/New_Home_Page/datafile/ctryprem.html",
]
RATING_URLS = [
    "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ratings.html",
    "https://pages.stern.nyu.edu/adamodar/New_Home_Page/datafile/ratings.html",
]
TE_URL = "https://tradingeconomics.com/bonds"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

DEFAULT_INDUSTRY = "Software (System & Application)"
DEFAULT_COUNTRY = "United States"
DEFAULT_ERP = 6.5      # %
DEFAULT_DEBT = 30.0    # %

# Placeholders used ONLY if a live fetch fails. They are flagged in the UI and
# should be replaced with manual overrides.
FALLBACK_BETAS = {DEFAULT_INDUSTRY: 1.20}
FALLBACK_TAX = {"United States": 0.25}
FALLBACK_SPREAD = 0.0125
FALLBACK_YIELDS = {"United States": 0.042}

# Country-name harmonisation between Damodaran and Trading Economics
ALIASES = {
    "us": "united states", "usa": "united states", "u.s.": "united states",
    "united states of america": "united states",
    "uk": "united kingdom", "great britain": "united kingdom",
    "south korea": "korea", "korea, republic of": "korea", "republic of korea": "korea",
    "czechia": "czech republic", "turkiye": "turkey", "türkiye": "turkey",
    "viet nam": "vietnam", "russian federation": "russia",
    "slovak republic": "slovakia", "hong kong sar": "hong kong",
    "ivory coast": "cote d'ivoire", "côte d'ivoire": "cote d'ivoire",
    "macedonia": "north macedonia", "euro area": "euro area",
}

CURRENCY = {
    "united states": "USD", "canada": "CAD", "mexico": "MXN", "brazil": "BRL",
    "chile": "CLP", "colombia": "COP", "peru": "PEN", "argentina": "ARS",
    "united kingdom": "GBP", "germany": "EUR", "france": "EUR", "italy": "EUR",
    "spain": "EUR", "netherlands": "EUR", "belgium": "EUR", "austria": "EUR",
    "portugal": "EUR", "ireland": "EUR", "greece": "EUR", "finland": "EUR",
    "slovakia": "EUR", "slovenia": "EUR", "croatia": "EUR", "lithuania": "EUR",
    "latvia": "EUR", "estonia": "EUR", "cyprus": "EUR", "malta": "EUR",
    "switzerland": "CHF", "sweden": "SEK", "norway": "NOK", "denmark": "DKK",
    "poland": "PLN", "czech republic": "CZK", "hungary": "HUF", "romania": "RON",
    "turkey": "TRY", "russia": "RUB", "israel": "ILS", "south africa": "ZAR",
    "egypt": "EGP", "nigeria": "NGN", "kenya": "KES", "morocco": "MAD",
    "japan": "JPY", "china": "CNY", "hong kong": "HKD", "taiwan": "TWD",
    "korea": "KRW", "india": "INR", "indonesia": "IDR", "malaysia": "MYR",
    "philippines": "PHP", "singapore": "SGD", "thailand": "THB",
    "vietnam": "VND", "pakistan": "PKR", "bangladesh": "BDT", "sri lanka": "LKR",
    "australia": "AUD", "new zealand": "NZD", "saudi arabia": "SAR",
    "qatar": "QAR", "united arab emirates": "AED", "serbia": "RSD",
    "bulgaria": "BGN", "iceland": "ISK", "ukraine": "UAH",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def norm_country(name: str) -> str:
    c = str(name).strip().lower()
    c = re.sub(r"\(.*?\)", "", c)
    c = re.sub(r"\s+", " ", c).strip()
    c = c.replace(" 10y", "").strip()
    return ALIASES.get(c, c)


def to_decimal(x) -> float | None:
    """Parse '21.00%', '1.11 %', 0.21, '4.25' into a decimal rate."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (int, float, np.number)):
        v = float(x)
        return v / 100 if abs(v) > 1.5 else v
    s = str(x).strip().replace(",", "")
    has_pct = "%" in s
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    v = float(m.group())
    if has_pct or abs(v) > 1.5:
        v /= 100
    return v


def http_get(urls: list[str], timeout: int = 25) -> tuple[requests.Response, str]:
    last_err = None
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r, url
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"All URLs failed: {last_err}")


def last_modified(r: requests.Response) -> str | None:
    lm = r.headers.get("Last-Modified")
    if not lm:
        return None
    try:
        return parsedate_to_datetime(lm).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return lm


DATE_RE = re.compile(
    r"(?:last\s+)?updated(?:\s+(?:in|on|as of))?\s*:?\s*"
    r"([A-Za-z]+\.?\s+\d{1,2},?\s+\d{4}|[A-Za-z]+\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4})",
    re.IGNORECASE,
)


def page_date(text: str) -> str | None:
    m = DATE_RE.search(text)
    return m.group(1).strip() if m else None


def now_str() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        " ".join(str(p) for p in c if "Unnamed" not in str(p)).strip()
        if isinstance(c, tuple) else str(c)
        for c in df.columns
    ]
    return df


def promote_header(df: pd.DataFrame, keyword: str) -> pd.DataFrame | None:
    """Return df whose header contains `keyword`, promoting a body row if needed."""
    df = flatten_cols(df)
    if any(keyword in c.lower() for c in df.columns):
        return df
    for i in range(min(8, len(df))):
        row = [str(v) for v in df.iloc[i].tolist()]
        if any(keyword in v.lower() for v in row):
            new = df.iloc[i + 1:].copy()
            new.columns = row
            return new
    return None


# --------------------------------------------------------------------------- #
# Data fetchers (cached; refresh button clears the cache)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_betas() -> tuple[pd.DataFrame, dict]:
    r, url = http_get(BETA_URLS)
    content = io.BytesIO(r.content)
    try:
        xls = pd.ExcelFile(content, engine="xlrd")
    except Exception:  # noqa: BLE001
        content.seek(0)
        xls = pd.ExcelFile(content, engine="openpyxl")

    sheet = next((s for s in xls.sheet_names if "industry" in s.lower()), xls.sheet_names[-1])
    raw = pd.read_excel(xls, sheet_name=sheet, header=None)

    # Locate header row ("Industry Name")
    hdr = None
    for i in range(min(40, len(raw))):
        if any(str(v).strip().lower() == "industry name" for v in raw.iloc[i].tolist()):
            hdr = i
            break
    if hdr is None:
        raise ValueError("Header 'Industry Name' not found in betas.xls")

    cols = [str(c).strip() for c in raw.iloc[hdr].tolist()]
    body = raw.iloc[hdr + 1:].reset_index(drop=True)
    body.columns = range(len(cols))
    low = [c.lower() for c in cols]

    i_name = low.index("industry name")
    i_beta = next((j for j, c in enumerate(low) if c == "unlevered beta"), None)
    if i_beta is None:
        i_beta = next(j for j, c in enumerate(low) if "unlevered beta" in c and "cash" not in c)

    df = pd.DataFrame({
        "Industry": body[i_name].astype(str).str.strip(),
        "Unlevered beta": pd.to_numeric(body[i_beta], errors="coerce"),
    }).dropna()
    df = df[(df["Industry"] != "nan") & (df["Industry"] != "")]
    df = df[~df["Industry"].str.lower().str.startswith("total market")]
    df = df.drop_duplicates("Industry").reset_index(drop=True)

    # Date stated inside the workbook
    date_txt = None
    for i in range(hdr):
        row = raw.iloc[i].tolist()
        for j, v in enumerate(row):
            if isinstance(v, str) and "date" in v.lower():
                for w in row[j + 1:]:
                    if pd.notna(w) and str(w).strip():
                        date_txt = (w.strftime("%Y-%m-%d") if hasattr(w, "strftime") else str(w))
                        break
            if date_txt:
                break
        if date_txt:
            break

    meta = {
        "url": url,
        "as_of": date_txt or last_modified(r) or "n/a",
        "fetched": now_str(),
        "note": f"Sheet '{sheet}', column '{cols[i_beta]}'",
    }
    return df, meta


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_tax() -> tuple[dict, dict]:
    r, url = http_get(TAX_URLS)
    tables = pd.read_html(io.StringIO(r.text))
    best = None
    for t in tables:
        t2 = promote_header(t, "tax")
        if t2 is not None and len(t2) > 20:
            best = t2
            break
    if best is None:
        raise ValueError("Corporate tax column not found in ctryprem.html")

    cols = [str(c) for c in best.columns]
    i_tax = next(j for j, c in enumerate(cols) if "tax" in c.lower())
    i_cty = next((j for j, c in enumerate(cols) if "country" in c.lower()), 0)

    out = {}
    for _, row in best.iterrows():
        name = str(row.iloc[i_cty]).strip()
        val = to_decimal(row.iloc[i_tax])
        if not name or name.lower() == "nan" or val is None:
            continue
        if 0 <= val < 1:
            out[name] = val
    if not out:
        raise ValueError("No tax rates parsed")

    meta = {
        "url": url,
        "as_of": page_date(BeautifulSoup(r.text, "lxml").get_text(" ")) or last_modified(r) or "n/a",
        "fetched": now_str(),
        "note": f"Column '{cols[i_tax]}'",
    }
    return out, meta


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_spread() -> tuple[float, dict]:
    r, url = http_get(RATING_URLS)
    soup = BeautifulSoup(r.text, "lxml")
    spread = None
    # First table on the page = large non-financial service firms.
    # Find the first row containing 'Baa2/BBB' and take the next % cell after it.
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        keys = [re.sub(r"\s+", "", c).lower() for c in cells]
        if "baa2/bbb" in keys:
            k = keys.index("baa2/bbb")
            for c in cells[k + 1:]:
                v = to_decimal(c)
                if v is not None and 0 < v < 0.2:
                    spread = v
                    break
        if spread is not None:
            break
    if spread is None:
        raise ValueError("Baa2/BBB spread not found in ratings.html")

    meta = {
        "url": url,
        "as_of": page_date(soup.get_text(" ")) or last_modified(r) or "n/a",
        "fetched": now_str(),
        "note": "Baa2/BBB, large non-financial service firms (first table)",
    }
    return spread, meta


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_yields() -> tuple[pd.DataFrame, dict]:
    r = requests.get(TE_URL, headers=HEADERS, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    rows, seen = [], set()
    for tr in soup.find_all("tr"):
        a = tr.find("a", href=True)
        if not a or "bond-yield" not in a["href"].lower():
            continue
        tds = tr.find_all("td")
        texts = [td.get_text(" ", strip=True) for td in tds]
        name = a.get_text(" ", strip=True)
        if not name:
            slug = a["href"].strip("/").split("/")[0]
            name = slug.replace("-", " ").title()
        name = re.sub(r"\s*10Y$", "", name).strip()
        # yield = first numeric cell after the name cell
        y = None
        start = next((i for i, t in enumerate(texts) if name.lower() in t.lower()), 0)
        for t in texts[start + 1:]:
            try:
                y = float(t.replace(",", "").replace("%", ""))
                break
            except ValueError:
                continue
        if y is None:
            continue
        key = norm_country(name)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"Country": name, "Yield": y / 100, "Date": texts[-1] if texts else ""})

    # Fallback: generic table parse
    if not rows:
        for t in pd.read_html(io.StringIO(r.text)):
            t = flatten_cols(t)
            ycol = next((c for c in t.columns if c.lower() == "yield"), None)
            if ycol is None:
                continue
            dcol = next((c for c in t.columns if c.lower() == "date"), None)
            for _, row in t.iterrows():
                name = re.sub(r"\s*10Y$", "", str(row.iloc[0])).strip()
                try:
                    y = float(row[ycol]) / 100
                except (TypeError, ValueError):
                    continue
                key = norm_country(name)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"Country": name, "Yield": y, "Date": str(row[dcol]) if dcol else ""})

    if not rows:
        raise ValueError("No yields parsed from Trading Economics")
    df = pd.DataFrame(rows)
    meta = {"url": TE_URL, "as_of": "per-country 'Date' column", "fetched": now_str(),
            "note": "10-year government bond yields"}
    return df, meta


def load_all():
    status = {}

    try:
        betas, m = fetch_betas(); status["betas"] = (True, m, None)
    except Exception as e:  # noqa: BLE001
        betas = pd.DataFrame({"Industry": list(FALLBACK_BETAS), "Unlevered beta": list(FALLBACK_BETAS.values())})
        status["betas"] = (False, {"url": BETA_URLS[0], "as_of": "placeholder", "fetched": now_str(), "note": ""}, str(e))

    try:
        tax, m = fetch_tax(); status["tax"] = (True, m, None)
    except Exception as e:  # noqa: BLE001
        tax = dict(FALLBACK_TAX)
        status["tax"] = (False, {"url": TAX_URLS[0], "as_of": "placeholder", "fetched": now_str(), "note": ""}, str(e))

    try:
        spread, m = fetch_spread(); status["spread"] = (True, m, None)
    except Exception as e:  # noqa: BLE001
        spread = FALLBACK_SPREAD
        status["spread"] = (False, {"url": RATING_URLS[0], "as_of": "placeholder", "fetched": now_str(), "note": ""}, str(e))

    try:
        yields, m = fetch_yields(); status["yields"] = (True, m, None)
    except Exception as e:  # noqa: BLE001
        yields = pd.DataFrame([{"Country": k, "Yield": v, "Date": "placeholder"} for k, v in FALLBACK_YIELDS.items()])
        status["yields"] = (False, {"url": TE_URL, "as_of": "placeholder", "fetched": now_str(), "note": ""}, str(e))

    return betas, tax, spread, yields, status


# --------------------------------------------------------------------------- #
# Core model
# --------------------------------------------------------------------------- #
def compute(beta_u, tax, w_d, erp, r_usd, r_fc, s_usd, s_fc_override=None) -> dict:
    """All inputs in decimals. Returns a dict of results in decimals."""
    de = w_d / (1 - w_d)
    beta_l = beta_u * (1 + (1 - tax) * de)
    s_fc_auto = s_usd * (1 + r_fc) / (1 + r_usd)
    s_fc = s_fc_override if s_fc_override is not None else s_fc_auto

    res = {"de": de, "beta_l": beta_l, "s_fc_auto": s_fc_auto, "s_fc": s_fc}
    for cur, r, s in (("USD", r_usd, s_usd), ("FC", r_fc, s_fc)):
        ke = r + beta_l * erp
        kd = r + s
        kd_after = kd * (1 - tax)
        wacc = (1 - w_d) * ke + w_d * kd_after
        res[cur] = {"rf": r, "spread": s, "ke": ke, "kd": kd, "kd_after": kd_after, "wacc": wacc}
    return res


def pct(x: float, d: int = 2) -> str:
    return f"{x * 100:.{d}f}%"


# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Data")
    if st.button("🔄 Refresh data", help="Clear the cache and re-fetch all sources"):
        st.cache_data.clear()
        st.rerun()

with st.spinner("Fetching data from Damodaran and Trading Economics…"):
    betas_df, tax_map, usd_spread_fetched, yields_df, status = load_all()

tax_norm = {norm_country(k): (k, v) for k, v in tax_map.items()}
yld_norm = {norm_country(r.Country): (r.Country, r.Yield, r.Date) for r in yields_df.itertuples()}

with st.sidebar:
    n_ok = sum(1 for v in status.values() if v[0])
    if n_ok == 4:
        st.success("All 4 sources loaded.")
    else:
        st.warning(f"{n_ok}/4 sources loaded. Failed sources use placeholders — use the manual overrides below.")

# Country list: Trading Economics countries first (they have yields), then Damodaran-only
country_names = {}
for k, (name, _, _) in yld_norm.items():
    country_names[k] = name
for k, (name, _) in tax_norm.items():
    country_names.setdefault(k, name)
country_names.setdefault("united states", "United States")
country_list = sorted(country_names.values(), key=str.lower)

industry_list = betas_df["Industry"].tolist()

# --------------------------------------------------------------------------- #
# Sidebar inputs
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Project inputs")
    ind_idx = next((i for i, s in enumerate(industry_list) if s.lower() == DEFAULT_INDUSTRY.lower()), 0)
    industry = st.selectbox("Industry", industry_list, index=ind_idx)

    cty_idx = next((i for i, s in enumerate(country_list) if norm_country(s) == "united states"), 0)
    country = st.selectbox("Target country", country_list, index=cty_idx)
    ckey = norm_country(country)
    ccy = CURRENCY.get(ckey, "local currency")

    erp = st.number_input("Equity risk premium (%)", 0.0, 20.0, DEFAULT_ERP, 0.1, format="%.2f") / 100
    w_d = st.slider("Target debt share D/(D+E) (%)", 0.0, 90.0, DEFAULT_DEBT, 1.0) / 100

    tax_choice = st.radio(
        "Project tax rate",
        ["U.S. corporate tax rate", f"{country} corporate tax rate"],
        index=0,
        help="The same tax rate is used for both currencies.",
    )

# Fetched values
beta_u_f = float(betas_df.loc[betas_df["Industry"] == industry, "Unlevered beta"].iloc[0])
us_tax_f = tax_norm.get("united states", (None, FALLBACK_TAX["United States"]))[1]
fc_tax_f = tax_norm.get(ckey, (None, None))[1]
us_y_f = yld_norm.get("united states", (None, FALLBACK_YIELDS["United States"], ""))[1]
fc_y_entry = yld_norm.get(ckey)
fc_y_f = fc_y_entry[1] if fc_y_entry else None

# --------------------------------------------------------------------------- #
# Manual overrides
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Manual overrides")
    st.caption("Tick a box to replace the fetched value.")

    def override(label, fetched, key, default_on=False, lo=-5.0, hi=100.0, step=0.01, is_pct=True):
        on = st.checkbox(f"Override {label}", value=default_on, key=f"on_{key}")
        shown = (fetched * 100 if is_pct else fetched) if fetched is not None else 0.0
        if on:
            v = st.number_input(
                f"{label}{' (%)' if is_pct else ''}", lo, hi, float(round(shown, 4)), step,
                format="%.4f" if not is_pct else "%.3f", key=f"val_{key}",
            )
            return (v / 100 if is_pct else v), True
        return fetched, False

    beta_u, ov_beta = override("unlevered beta", beta_u_f, f"beta_{industry}", lo=0.0, hi=5.0, is_pct=False)
    us_tax, ov_ustax = override("U.S. corporate tax rate", us_tax_f, "ustax", lo=0.0, hi=60.0)
    fc_tax, ov_fctax = override(
        f"{country} corporate tax rate", fc_tax_f if fc_tax_f is not None else us_tax,
        f"fctax_{ckey}", default_on=fc_tax_f is None, lo=0.0, hi=60.0,
    )
    s_usd, ov_susd = override("USD BBB spread", usd_spread_fetched, "susd", lo=0.0, hi=20.0)
    r_usd, ov_rusd = override("U.S. 10Y yield", us_y_f, "rusd", lo=-2.0, hi=50.0)
    r_fc, ov_rfc = override(
        f"{country} 10Y yield", fc_y_f if fc_y_f is not None else r_usd,
        f"rfc_{ckey}", default_on=fc_y_f is None, lo=-2.0, hi=100.0,
    )

    # Keep the U.S. as one country: if the target is the U.S., FC yield/tax follow U.S. unless overridden
    if ckey == "united states":
        if not ov_rfc:
            r_fc = r_usd
        if not ov_fctax:
            fc_tax = us_tax

    st.markdown("---")
    ov_sfc = st.checkbox(f"Override {ccy} BBB spread", value=False, key="on_sfc")
    auto_preview = s_usd * (1 + r_fc) / (1 + r_usd)
    if ov_sfc:
        s_fc_manual = st.number_input(
            f"{ccy} BBB spread (%)", 0.0, 50.0, float(round(auto_preview * 100, 4)), 0.01,
            format="%.3f", key=f"val_sfc_{ckey}",
        ) / 100
    else:
        s_fc_manual = None
        st.caption(f"Auto-calculated {ccy} spread: **{pct(auto_preview, 3)}**")

if fc_y_f is None:
    st.warning(f"No Trading Economics 10Y yield found for **{country}**. Enter it under *Manual overrides*.")
if fc_tax_f is None:
    st.warning(f"No Damodaran corporate tax rate found for **{country}**. Enter it under *Manual overrides*.")

T = us_tax if tax_choice.startswith("U.S.") else fc_tax
res = compute(beta_u, T, w_d, erp, r_usd, r_fc, s_usd, s_fc_manual)

# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title("International cost of capital")
st.write(
    f"**{industry}** project, cost of capital in **USD** and in **{country}'s currency ({ccy})**, "
    f"using one project beta, ERP, tax rate and capital structure."
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Unlevered beta", f"{beta_u:.3f}", "manual" if ov_beta else None, delta_color="off")
c2.metric("Levered beta", f"{res['beta_l']:.3f}")
c3.metric("D/E", f"{res['de']:.3f}")
c4.metric("Project tax rate", pct(T), tax_choice.split(" corporate")[0], delta_color="off")
c5.metric("ERP", pct(erp))

st.subheader("Results side by side")
left, right = st.columns(2)
for col, cur, label in ((left, "USD", "USD (United States)"), (right, "FC", f"{ccy} ({country})")):
    r = res[cur]
    with col:
        st.markdown(f"#### {label}")
        a, b = st.columns(2)
        a.metric("10Y government yield", pct(r["rf"]))
        b.metric("BBB spread", pct(r["spread"], 3))
        a.metric("Cost of equity", pct(r["ke"]))
        b.metric("Pre-tax cost of debt", pct(r["kd"]))
        a.metric("After-tax cost of debt", pct(r["kd_after"]))
        b.metric("WACC", pct(r["wacc"]))

u, f = res["USD"], res["FC"]
summary = pd.DataFrame(
    {
        "USD": [u["rf"], u["spread"], res["beta_l"], u["ke"], u["kd"], u["kd_after"], u["wacc"]],
        ccy if ccy != "USD" else "USD (target)": [f["rf"], f["spread"], res["beta_l"], f["ke"], f["kd"], f["kd_after"], f["wacc"]],
    },
    index=["10Y yield", "BBB spread", "Levered beta", "Cost of equity",
           "Pre-tax cost of debt", "After-tax cost of debt", "WACC"],
)
summary["Difference (target − USD)"] = summary.iloc[:, 1] - summary.iloc[:, 0]
disp = summary.astype(object)
for idx in disp.index:
    if idx == "Levered beta":
        disp.loc[idx] = [f"{v:.3f}" for v in summary.loc[idx]]
    else:
        disp.loc[idx] = [f"{v * 100:.3f}%" for v in summary.loc[idx]]
st.table(disp)

chart_df = pd.DataFrame(
    {"USD": [u["ke"] * 100, u["wacc"] * 100], ccy if ccy != "USD" else "USD (target)": [f["ke"] * 100, f["wacc"] * 100]},
    index=["Cost of equity (%)", "WACC (%)"],
)
st.bar_chart(chart_df)

# Step-by-step calculation (for independent checking)
with st.expander("Step-by-step calculation (use this to check by hand)"):
    st.markdown(
        f"""
1. D/E = {w_d:.4f} / (1 − {w_d:.4f}) = **{res['de']:.4f}**
2. β_L = {beta_u:.4f} × [1 + (1 − {T:.4f}) × {res['de']:.4f}] = **{res['beta_l']:.4f}**
3. k_E,USD = {u['rf']:.4f} + {res['beta_l']:.4f} × {erp:.4f} = **{u['ke']:.4f}**
4. k_E,{ccy} = {f['rf']:.4f} + {res['beta_l']:.4f} × {erp:.4f} = **{f['ke']:.4f}**
5. s_{ccy} (auto) = {s_usd:.4f} × (1 + {r_fc:.4f}) / (1 + {r_usd:.4f}) = **{res['s_fc_auto']:.5f}**{"  — overridden to **" + f"{res['s_fc']:.5f}" + "**" if s_fc_manual is not None else ""}
6. k_D,USD = {u['rf']:.4f} + {u['spread']:.4f} = **{u['kd']:.4f}**; k_D,{ccy} = {f['rf']:.4f} + {f['spread']:.5f} = **{f['kd']:.5f}**
7. WACC_USD = (1 − {w_d:.4f}) × {u['ke']:.4f} + {w_d:.4f} × {u['kd']:.4f} × (1 − {T:.4f}) = **{u['wacc']:.5f}**
8. WACC_{ccy} = (1 − {w_d:.4f}) × {f['ke']:.4f} + {w_d:.4f} × {f['kd']:.5f} × (1 − {T:.4f}) = **{f['wacc']:.5f}**
"""
    )

# Assumptions
with st.expander("Model equations and assumptions", expanded=False):
    st.latex(r"\beta_L=\beta_U\left[1+(1-T)\frac{D}{E}\right],\qquad \frac{D}{E}=\frac{w_D}{1-w_D}")
    st.latex(r"k_{E,c}=r_{10Y,c}+\beta_L\,ERP")
    st.latex(r"s^{auto}_{FC}=s_{USD}\,\frac{1+r_{10Y,FC}}{1+r_{10Y,USD}}\quad\text{(rates in decimals)}")
    st.latex(r"k_{D,c}=r_{10Y,c}+s_c,\qquad WACC_c=(1-w_D)\,k_{E,c}+w_D\,k_{D,c}\,(1-T)")
    st.markdown(
        """
**Interpretation.** Both currencies use the same project beta, ERP, tax rate and debt/equity weights.
Each currency's base rate is proxied by its 10-year government bond yield. The automatic spread rule
is a modeling convention that keeps the USD debt premium after a gross-rate currency translation; it is
not an observed local-currency BBB spread. Local sovereign risk, hedging costs, country risk premia and
cross-currency basis are **not** modeled separately.

* **Beta**: Damodaran U.S. industry *unlevered* beta, relevered with the Hamada equation above.
* **Tax rate**: one project tax rate (U.S. or target country), used in relevering and in the debt tax shield.
* **USD spread**: Damodaran's Baa2/BBB default spread for large non-financial service firms.
* **Yields**: latest 10-year government yields on Trading Economics (may be a few minutes delayed).
"""
    )

# Verification checks
with st.expander("Built-in verification checks", expanded=False):
    tol = 1e-12
    eq = compute(beta_u, T, w_d, erp, r_usd, r_usd, s_usd, None)
    chk1 = all(abs(eq["USD"][k] - eq["FC"][k]) < tol for k in ("ke", "kd", "wacc"))
    z = compute(beta_u, T, 0.0, erp, r_usd, r_fc, s_usd, s_fc_manual)
    chk2 = abs(z["USD"]["wacc"] - z["USD"]["ke"]) < tol and abs(z["FC"]["wacc"] - z["FC"]["ke"]) < tol
    base = compute(beta_u, T, w_d, erp, r_usd, r_fc, s_usd, None)
    ovr = compute(beta_u, T, w_d, erp, r_usd, r_fc, s_usd, base["s_fc_auto"] + 0.01)
    chk3 = all(abs(base["USD"][k] - ovr["USD"][k]) < tol for k in ("ke", "kd", "kd_after", "wacc"))
    st.dataframe(
        pd.DataFrame(
            {
                "Check": [
                    "Equal 10Y yields + spread override off → identical USD and FC results",
                    "Zero debt → WACC equals cost of equity (both currencies)",
                    "Overriding the FC spread (+1pp) leaves USD results unchanged",
                ],
                "Result": ["✅ Pass" if c else "❌ Fail" for c in (chk1, chk2, chk3)],
                "Detail": [
                    f"WACC USD {pct(eq['USD']['wacc'], 4)} vs FC {pct(eq['FC']['wacc'], 4)}",
                    f"USD: WACC {pct(z['USD']['wacc'], 4)} = kE {pct(z['USD']['ke'], 4)}",
                    f"USD WACC {pct(base['USD']['wacc'], 4)} → {pct(ovr['USD']['wacc'], 4)}",
                ],
            }
        ),
        hide_index=True,
    )

# --------------------------------------------------------------------------- #
# All-industry comparison
# --------------------------------------------------------------------------- #
st.subheader("All-industry comparison")
st.caption(
    f"Every Damodaran industry with the current settings: {country}, ERP {pct(erp)}, debt {pct(w_d, 0)}, "
    f"tax {pct(T)}. Manual beta override applies only to the selected industry."
)
rows = []
for name, bu in zip(betas_df["Industry"], betas_df["Unlevered beta"]):
    bu_used = beta_u if name == industry else float(bu)
    rr = compute(bu_used, T, w_d, erp, r_usd, r_fc, s_usd, s_fc_manual)
    rows.append({
        "Industry": name,
        "Unlevered beta": bu_used,
        "Levered beta": rr["beta_l"],
        "Cost of equity USD (%)": rr["USD"]["ke"] * 100,
        f"Cost of equity {ccy} (%)": rr["FC"]["ke"] * 100,
        "WACC USD (%)": rr["USD"]["wacc"] * 100,
        f"WACC {ccy} (%)": rr["FC"]["wacc"] * 100,
    })
all_df = pd.DataFrame(rows)
if ccy == "USD":
    all_df = all_df.loc[:, ~all_df.columns.duplicated()]
all_df = all_df.sort_values("WACC USD (%)", ascending=False).reset_index(drop=True)

rank = int(all_df.index[all_df["Industry"] == industry][0]) + 1
st.write(f"**{industry}** ranks **#{rank} of {len(all_df)}** by USD WACC (highest first).")
st.dataframe(
    all_df.style.format({c: "{:.3f}" for c in all_df.columns if c != "Industry"})
    .apply(lambda r: ["background-color: rgba(255, 200, 0, 0.25)" if r["Industry"] == industry else "" for _ in r], axis=1),
    hide_index=True,
    height=420,
)

# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #
st.subheader("Download results")
inputs_df = pd.DataFrame(
    {
        "Parameter": [
            "Industry", "Target country", "Target currency", "Unlevered beta", "Levered beta",
            "ERP", "Debt share", "D/E", "Project tax rate source", "Project tax rate",
            "U.S. 10Y yield", f"{country} 10Y yield", "USD BBB spread",
            f"{ccy} BBB spread (used)", f"{ccy} BBB spread (auto)", f"{ccy} spread overridden",
        ],
        "Value": [
            industry, country, ccy, beta_u, res["beta_l"], erp, w_d, res["de"], tax_choice, T,
            r_usd, r_fc, s_usd, res["s_fc"], res["s_fc_auto"], s_fc_manual is not None,
        ],
    }
)
results_df = summary.reset_index().rename(columns={"index": "Item"})
sources_rows = []
for key, label in (("betas", "Unlevered industry betas"), ("tax", "Corporate tax rates"),
                   ("spread", "USD BBB spread"), ("yields", "10Y government yields")):
    ok, m, err = status[key]
    sources_rows.append({
        "Data": label, "Status": "live" if ok else f"FAILED – placeholder ({err})",
        "Source": m["url"], "As of": m["as_of"], "Fetched": m["fetched"], "Note": m.get("note", ""),
    })
sources_df = pd.DataFrame(sources_rows)

buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as xw:
    inputs_df.to_excel(xw, sheet_name="Inputs", index=False)
    results_df.to_excel(xw, sheet_name="Results", index=False)
    all_df.to_excel(xw, sheet_name="All industries", index=False)
    sources_df.to_excel(xw, sheet_name="Sources", index=False)

d1, d2, d3 = st.columns(3)
d1.download_button("Download scenario (CSV)",
                   pd.concat([inputs_df.astype(str), results_df.astype(str)], axis=0).to_csv(index=False),
                   "cost_of_capital_scenario.csv", "text/csv")
d2.download_button("Download all industries (CSV)", all_df.to_csv(index=False),
                   "cost_of_capital_all_industries.csv", "text/csv")
d3.download_button("Download full workbook (Excel)", buf.getvalue(), "cost_of_capital.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------------------------------------------------------------------------- #
# Sources and dates
# --------------------------------------------------------------------------- #
st.subheader("Data sources and dates")
used = {
    "betas": f"β_U = {beta_u_f:.4f} ({industry})",
    "tax": f"U.S. {pct(us_tax_f)}" + (f"; {country} {pct(fc_tax_f)}" if fc_tax_f is not None else f"; {country} n/a"),
    "spread": f"{pct(usd_spread_fetched, 3)}",
    "yields": f"U.S. {pct(us_y_f, 3)} ({yld_norm.get('united states', ('', '', ''))[2]})"
              + (f"; {country} {pct(fc_y_f, 3)} ({fc_y_entry[2]})" if fc_y_entry else f"; {country} n/a"),
}
src_show = sources_df.copy()
src_show.insert(2, "Fetched value", [used[k] for k in ("betas", "tax", "spread", "yields")])
st.dataframe(src_show, hide_index=True)

overrides = [n for n, on in (("unlevered beta", ov_beta), ("U.S. tax", ov_ustax), (f"{country} tax", ov_fctax),
                             ("USD spread", ov_susd), ("U.S. yield", ov_rusd), (f"{country} yield", ov_rfc),
                             (f"{ccy} spread", s_fc_manual is not None)) if on]
if overrides:
    st.info("Manual overrides in use: " + ", ".join(overrides))

with st.expander("All fetched 10Y yields"):
    st.dataframe(yields_df.assign(**{"Yield (%)": yields_df["Yield"] * 100}).drop(columns="Yield"), hide_index=True)

st.caption("Educational tool. Data © Aswath Damodaran (NYU Stern) and Trading Economics. Verify before use.")
