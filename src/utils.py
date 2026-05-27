"""
Shared statistical helpers for the AGYW teenage pregnancy analysis.
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Prevalence & confidence intervals
# ---------------------------------------------------------------------------

def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for a proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    cen = p + z**2 / (2 * n)
    hw = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lo, hi = (cen - hw) / den, (cen + hw) / den
    return max(0.0, lo), min(1.0, hi)


def add_prev_ci(df, num='cases', den='n'):
    """Append Wilson CI columns and percentage-formatted prevalence to a counts df."""
    out = df.copy()
    p = out[num] / out[den]
    lo_hi = out.apply(
        lambda r: pd.Series(wilson_ci(int(r[num]), int(r[den])), index=['ci_low', 'ci_high']),
        axis=1,
    )
    out = pd.concat([out, lo_hi], axis=1)
    out['prev_pct'] = (100 * p).round(1)
    out['ci_low_pct'] = (100 * out['ci_low']).round(1)
    out['ci_high_pct'] = (100 * out['ci_high']).round(1)
    return out


def mean_sd(x):
    """Return (mean, std) of a series."""
    return x.mean(), x.std(ddof=1)


# ---------------------------------------------------------------------------
# Effect-size helpers
# ---------------------------------------------------------------------------

def cramers_v(chi2, n, r, c):
    """Bias-corrected Cramér's V from a chi-square statistic."""
    phi2 = chi2 / n
    k = min(r - 1, c - 1)
    if k == 0 or n == 0:
        return np.nan
    phi2corr = max(0, phi2 - ((c - 1) * (r - 1)) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    ccorr = c - (c - 1) ** 2 / (n - 1)
    return np.sqrt(phi2corr / max(1e-12, min(rcorr - 1, ccorr - 1)))


def pformat(p):
    """Format a p-value string: '<0.001' or 3 decimal places."""
    if pd.isna(p):
        return np.nan
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------------------
# Descriptive table builder
# ---------------------------------------------------------------------------

def build_block_chi2(
    df,
    var,
    *,
    categories=None,
    label_map=None,
    drop_values=None,
    tidy_name=None,
    exclude_missing=False,
    add_effect_size=True,
):
    """
    Build one characteristic block for a descriptive table.

    Returns a DataFrame with counts for Total / Ever-pregnant / Never-pregnant,
    one omnibus p-value (Fisher for 2×2, chi-square otherwise), and optional
    Cramér's V effect size.
    """
    x = df.copy()
    s = x[var].copy()

    if exclude_missing:
        x = x[s.notna()].copy()
        s = x[var]

    if label_map is not None:
        s = s.map(label_map).fillna(s)

    if drop_values is not None:
        keep = ~s.isin(drop_values)
        x = x[keep].copy()
        s = s[keep]

    cats = pd.Index(categories) if categories is not None else pd.Index(pd.Series(s).dropna().unique())
    x = x.assign(_cat=s)
    x['_cat'] = pd.Categorical(x['_cat'], categories=cats, ordered=True)

    tot   = x['_cat'].value_counts().reindex(cats, fill_value=0)
    ever  = x.loc[x['ado_preg'] == 1, '_cat'].value_counts().reindex(cats, fill_value=0)
    never = x.loc[x['ado_preg'] == 0, '_cat'].value_counts().reindex(cats, fill_value=0)

    block = pd.DataFrame({
        tidy_name or var: cats,
        'Total (n)':               tot.astype('Int64'),
        'Ever-been pregnant (n)':  ever.astype('Int64'),
        'Never been pregnant (n)': never.astype('Int64'),
    })

    ct = pd.crosstab(
        x['ado_preg'].map({0: 'Never', 1: 'Ever'}),
        x['_cat'],
    ).reindex(index=['Never', 'Ever'], fill_value=0)
    ct = ct.loc[:, ct.sum(0) > 0]

    p = np.nan
    V = np.nan
    if ct.shape == (2, 2):
        _, p = stats.fisher_exact(ct.values)
        chi2, _, _, _ = stats.chi2_contingency(ct.values, correction=False)
        V = cramers_v(chi2, n=int(ct.values.sum()), r=2, c=2) if add_effect_size else np.nan
    elif ct.shape[0] == 2 and ct.shape[1] >= 2:
        chi2, p, _, _ = stats.chi2_contingency(ct.values, correction=False)
        V = cramers_v(chi2, n=int(ct.values.sum()), r=ct.shape[0], c=ct.shape[1]) if add_effect_size else np.nan

    pcol = [np.nan] * len(block)
    Vcol = [np.nan] * len(block)
    if len(block) > 0:
        pcol[0] = p
        Vcol[0] = V
    block['p-value'] = [pformat(v) for v in pcol]
    if add_effect_size:
        block["Cramér's V"] = Vcol

    return block


# ---------------------------------------------------------------------------
# Survival analysis helper
# ---------------------------------------------------------------------------

def km_incidence_at(kmf, t):
    """
    Return cumulative incidence (and 95% CI) at time t from a fitted
    KaplanMeierFitter object.
    """
    sf  = kmf.survival_function_
    ci  = kmf.confidence_interval_
    sf_col   = sf.columns[0]
    low_col  = [c for c in ci.columns if 'lower' in c.lower()][0]
    high_col = [c for c in ci.columns if 'upper' in c.lower()][0]
    i_sf = max(0, min(len(sf) - 1, sf.index.searchsorted(t, side='right') - 1))
    i_ci = max(0, min(len(ci) - 1, ci.index.searchsorted(t, side='right') - 1))
    S    = float(sf.iloc[i_sf][sf_col])
    S_lo = float(ci.iloc[i_ci][low_col])
    S_hi = float(ci.iloc[i_ci][high_col])
    return 1 - S, 1 - S_hi, 1 - S_lo  # invert CI correctly


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

def or_ci(res, var):
    """Return (OR, CI_low, CI_high, p) for a single term from a fitted Logit result."""
    b = res.params[var]
    lo, hi = res.conf_int().loc[var]
    return float(np.exp(b)), float(np.exp(lo)), float(np.exp(hi)), float(res.pvalues[var])


def tidy_or_table(result, z=1.96):
    """Convert a fitted statsmodels Logit result to a tidy OR DataFrame."""
    rows = []
    for term, b in result.params.items():
        if term == 'const':
            continue
        se = result.bse.get(term, np.nan)
        lo = b - z * se if np.isfinite(se) else np.nan
        hi = b + z * se if np.isfinite(se) else np.nan
        rows.append({
            'term':    term,
            'OR':      np.exp(b),
            'CI_low':  np.exp(lo) if np.isfinite(lo) else np.nan,
            'CI_high': np.exp(hi) if np.isfinite(hi) else np.nan,
            'p':       result.pvalues.get(term, np.nan),
        })
    return pd.DataFrame(rows)


def fit_model(sub):
    """
    Fit a logistic regression of ado_preg on proximal determinants for a
    year-stratified sub-sample. Returns a fitted statsmodels Logit result.
    """
    sub = sub.copy()
    pg = sub['person_sex_group'].astype('string').str.strip().str.lower()
    sub['partner_show'] = np.where(pg.isin(['husband', 'boyfriend']), pg, 'other')
    partner = pd.get_dummies(sub['partner_show'], prefix='partner', drop_first=True, dtype=float)

    cols = ['ado_preg', 'sex_age_teen', 'marry_age_teen',
            'do_anything_binary', 'will_sex_binary', 'age_completed']
    dm = sub[cols].join(partner)
    dm = dm.apply(pd.to_numeric, errors='coerce').dropna()

    y = dm['ado_preg'].astype(int)
    X = sm.add_constant(dm.drop(columns=['ado_preg']).astype(float))
    return sm.Logit(y, X).fit(disp=False)


# ---------------------------------------------------------------------------
# Method-use summary
# ---------------------------------------------------------------------------

def summarize_methods(df, method_cols):
    """
    Return (n_users, table) where table shows n_yes and pct_of_users per method.
    """
    n_users = len(df)
    if n_users == 0:
        empty = pd.DataFrame({
            'n_yes':        pd.Series(dtype='int64'),
            'pct_of_users': pd.Series(dtype='float'),
        })
        return 0, empty
    counts = df[method_cols].sum(axis=0).astype(int)
    pct = (counts / n_users * 100).round(1)
    tab = pd.DataFrame({'n_yes': counts, 'pct_of_users': pct}).sort_values('pct_of_users', ascending=False)
    return int(n_users), tab
