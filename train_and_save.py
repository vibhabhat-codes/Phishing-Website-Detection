"""
=============================================================================
 PHISHING DETECTION — Train & Save All Model Artifacts
 Run this ONCE before starting the web server.

 Usage:
   pip install pandas openpyxl scikit-learn lightgbm joblib
   python train_and_save.py

 Output files (saved to ./model/):
   model_artifacts.pkl  — stacked ensemble + imputer + scaler + feature lists
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_LGB = False
    print("[INFO] LightGBM not found — using GradientBoostingClassifier.")

SEED = 42
np.random.seed(SEED)

# ── Paths — update these to point to your xlsx files ─────────────────────────
FEATURES_FILE = "All_Features_threshold90.xlsx"
MAPPING_FILE  = "Mapping_File.xlsx"
MODEL_DIR     = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load Dataset ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  PHISHING DETECTION — TRAINING PIPELINE")
print("=" * 60)

df = pd.read_excel(FEATURES_FILE)
print(f"\n[OK] Loaded {df.shape[0]} rows × {df.shape[1]} columns")
print(f"     Legitimate: {(df.label==0).sum()} | Phishing: {(df.label==1).sum()}")

# ── Feature Groups ────────────────────────────────────────────────────────────
URL_FEATURES = [
    'brand_subdomain','brand_substring_domain','brand_typo_dom',
    'fake_tld','brand_in_path_query','dots_path',
    'url_more_than_mean','domain_more_than_mean','gibberish',
    'domain_length','url_length','url_contains_ip',
    'presence_of_shortened_url_targeturl','special_characters_count',
    'no_of_at','no_of_comma','no_of_dollar','no_of_semicolon',
    'no_of_space','no_of_ampersand','no_of_dots','no_of_equal',
    'no_of_percent','no_of_underscore','no_of_hyphen',
    'no_of_questionmark','no_of_colon','no_of_slashes_inpath',
    'has_asterisk','has_or','has_embedded_url','uses_https',
    'num_query_params','numeric_proportion','url_entropy',
    'hostname_digit_ratio','has_redirections','has_punycode',
    'count_www','count_com','url_contains_risky_extension',
    'has_explicit_port','total_words_url','average_length_of_words',
    'longest_word_length','shortest_word_length','presence_of_free_hosting',
]

HTML_FEATURES = [
    'no_of_forms','no_of_Images','no_of_hyperlinks',
    'no_of_external_links','no_of_internal_links','no_of_null_links',
    'no_of_selfreference_links','no_of_empty_links',
    'RatioOfEmptyLinksToExternalLinks',
    'htmlcontent_contains_risky_extensions',
    'presence_of_shorten_url_html','hidden_elements_html',
    'is_script_loaded_from_external_domain','suspicious_inline_js',
    'is_footer_mismatch','is_header_mismatch','is_copyright_mismatch',
    'is_socials_mismatch','is_favicon_mismatch','has_missing_title',
    'title_mismatch_with_domain','has_inline_event_handlers',
    'form_action_external',
]

URL_FEATURES  = [c for c in URL_FEATURES  if c in df.columns]
HTML_FEATURES = [c for c in HTML_FEATURES if c in df.columns]
print(f"\n[OK] URL features  : {len(URL_FEATURES)}")
print(f"[OK] HTML features : {len(HTML_FEATURES)}")

# ── Suspicion Score ───────────────────────────────────────────────────────────
def compute_suspicion_score(df):
    s = pd.DataFrame(index=df.index)
    if 'url_entropy' in df.columns:
        mn, mx = df['url_entropy'].min(), df['url_entropy'].max()
        s['s1'] = (df['url_entropy'] - mn) / (mx - mn + 1e-9)
    else:
        s['s1'] = 0.0

    spc_cols = ['no_of_at','no_of_percent','no_of_hyphen',
                'no_of_underscore','no_of_semicolon','no_of_dollar']
    present  = [c for c in spc_cols if c in df.columns]
    if present and 'url_length' in df.columns:
        density  = df[present].sum(axis=1) / (df['url_length'].replace(0,1))
        mn, mx   = density.min(), density.max()
        s['s2']  = (density - mn) / (mx - mn + 1e-9)
    else:
        s['s2'] = 0.0

    if 'domain_length' in df.columns:
        mn, mx  = df['domain_length'].min(), df['domain_length'].max()
        s['s3'] = (df['domain_length'] - mn) / (mx - mn + 1e-9)
    else:
        s['s3'] = 0.0

    if 'no_of_external_links' in df.columns and 'no_of_internal_links' in df.columns:
        total   = df['no_of_external_links'] + df['no_of_internal_links'] + 1
        ratio   = df['no_of_external_links'] / total
        mn, mx  = ratio.min(), ratio.max()
        s['s4'] = (ratio - mn) / (mx - mn + 1e-9)
    else:
        s['s4'] = 0.0

    return s[['s1','s2','s3','s4']].mean(axis=1)

df['suspicion_score'] = compute_suspicion_score(df)
print(f"\n[★] Suspicion Score engineered")
print(f"    Phishing mean  : {df[df.label==1]['suspicion_score'].mean():.4f}")
print(f"    Legitimate mean: {df[df.label==0]['suspicion_score'].mean():.4f}")

ALL_FEATURES = URL_FEATURES + HTML_FEATURES + ['suspicion_score']

# Save entropy stats for inference-time normalization
ENTROPY_STATS = {
    'url_entropy_min': df['url_entropy'].min() if 'url_entropy' in df.columns else 0,
    'url_entropy_max': df['url_entropy'].max() if 'url_entropy' in df.columns else 1,
    'domain_length_min': df['domain_length'].min() if 'domain_length' in df.columns else 0,
    'domain_length_max': df['domain_length'].max() if 'domain_length' in df.columns else 1,
}

spc_cols = ['no_of_at','no_of_percent','no_of_hyphen','no_of_underscore','no_of_semicolon','no_of_dollar']
present  = [c for c in spc_cols if c in df.columns]
if present and 'url_length' in df.columns:
    density = df[present].sum(axis=1) / (df['url_length'].replace(0,1))
    ENTROPY_STATS['spc_density_min'] = density.min()
    ENTROPY_STATS['spc_density_max'] = density.max()

if 'no_of_external_links' in df.columns and 'no_of_internal_links' in df.columns:
    total = df['no_of_external_links'] + df['no_of_internal_links'] + 1
    ratio = df['no_of_external_links'] / total
    ENTROPY_STATS['ext_ratio_min'] = ratio.min()
    ENTROPY_STATS['ext_ratio_max'] = ratio.max()

# ── Prepare X, y ──────────────────────────────────────────────────────────────
X = df[ALL_FEATURES].copy()
y = df['label'].astype(int)

imputer   = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=ALL_FEATURES)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=SEED, stratify=y
)

scaler     = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=ALL_FEATURES)
X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=ALL_FEATURES)

print(f"\n[OK] Train: {len(X_train_sc)} | Test: {len(X_test_sc)}")

# ── Build & Train ─────────────────────────────────────────────────────────────
if HAS_LGB:
    url_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        colsample_bytree=0.8, subsample=0.8, random_state=SEED, verbose=-1
    )
else:
    url_model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5, random_state=SEED
    )

html_model = RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=5,
    random_state=SEED, n_jobs=-1
)
meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)

stacked = StackingClassifier(
    estimators=[('url_stream', url_model), ('html_stream', html_model)],
    final_estimator=meta_model,
    cv=5, n_jobs=-1
)

print("\n[Training] Stacked Ensemble (may take ~1-2 min)…")
stacked.fit(X_train_sc, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = stacked.predict(X_test_sc)
y_prob = stacked.predict_proba(X_test_sc)[:, 1]
auc    = roc_auc_score(y_test, y_prob)
ap     = average_precision_score(y_test, y_prob)

print(f"\n[Results] Test Set Performance")
print(classification_report(y_test, y_pred, target_names=["Legitimate","Phishing"]))
print(f"ROC-AUC : {auc:.4f}")
print(f"Avg-Prec: {ap:.4f}")

# ── Save Artifacts ────────────────────────────────────────────────────────────
artifacts = {
    'model':         stacked,
    'imputer':       imputer,
    'scaler':        scaler,
    'url_features':  URL_FEATURES,
    'html_features': HTML_FEATURES,
    'all_features':  ALL_FEATURES,
    'entropy_stats': ENTROPY_STATS,
}

out_path = os.path.join(MODEL_DIR, "model_artifacts.pkl")
joblib.dump(artifacts, out_path)
print(f"\n[✓] Artifacts saved → {out_path}")
print("[DONE] You can now start the Flask server: python app.py")
