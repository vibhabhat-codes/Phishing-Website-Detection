"""
=============================================================================
 PHISHING DETECTION — Flask Backend (v3 — Analytics + Explainability)

 Fixes applied (v2):
   1. HTML fetch failure → use dataset medians (not zeros)
   2. Dual-threshold: prob > 0.65 AND suspicion_score > 0.28 for phishing
   3. Known legitimate domain allowlist (pinterest, google, etc.)
   4. Better user-agent & redirect handling

 New in v3 (additive only — no existing behavior changed):
   1. SQLite scan logging (scans.db) → powers the analytics dashboard
   2. GET  /dashboard/stats        → aggregate stats for charts
   3. SHAP-based "why was this flagged" top contributing features
   4. POST /predict/batch          → CSV upload or JSON list of URLs
   5. Redirect chain capture (resp.history) surfaced in /predict response
   6. Optional Google Safe Browsing cross-check (set GOOGLE_SAFE_BROWSING_API_KEY)
   7. POST /feedback               → thumbs up/down per scan_id

 Usage:
   pip install -r requirements.txt
   python app.py
=============================================================================
"""

import re
import os
import io
import csv
import math
import time
import json
import sqlite3
import datetime
import urllib.parse
from collections import Counter

import numpy as np
import pandas as pd
import joblib
import requests
import urllib3
urllib3.disable_warnings()

from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[INFO] shap not installed — /predict will skip 'why flagged' explanations.")

app = Flask(__name__)
CORS(app)

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "scans.db")
GOOGLE_SAFE_BROWSING_API_KEY = os.environ.get("GOOGLE_SAFE_BROWSING_API_KEY", "").strip()

# ── Load model artifacts ──────────────────────────────────────────────────────
print("[Boot] Loading model artifacts…")
artifacts     = joblib.load("model/model_artifacts.pkl")
model         = artifacts['model']
imputer       = artifacts['imputer']
scaler        = artifacts['scaler']
URL_FEATURES  = artifacts['url_features']
HTML_FEATURES = artifacts['html_features']
ALL_FEATURES  = artifacts['all_features']
ESTATS        = artifacts['entropy_stats']
print("[Boot] Model ready.")

# ── SHAP explainer (lazy — built on first actual use, not at boot) ──────────
# Building this at boot delayed startup and added memory pressure right when the
# process is most vulnerable to being killed by a host's startup/health-check
# timeout. Deferring it to the first real prediction spreads that cost out and
# means a slow/failed build never blocks the server from coming up at all.
SHAP_EXPLAINER = None
SHAP_EXPLAINER_ATTEMPTED = False

def get_shap_explainer():
    global SHAP_EXPLAINER, SHAP_EXPLAINER_ATTEMPTED
    if SHAP_EXPLAINER_ATTEMPTED:
        return SHAP_EXPLAINER
    SHAP_EXPLAINER_ATTEMPTED = True
    if not HAS_SHAP:
        return None
    try:
        # The LightGBM "url_stream" base learner is fast and TreeExplainer-friendly.
        SHAP_EXPLAINER = shap.TreeExplainer(model.named_estimators_['url_stream'])
        print("[Info] SHAP explainer built on first use.")
    except Exception as e:
        print(f"[WARN] Could not build SHAP explainer, disabling explanations: {e}")
        SHAP_EXPLAINER = None
    return SHAP_EXPLAINER

# ── SQLite: scan log + feedback (powers the analytics dashboard) ────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            url               TEXT NOT NULL,
            prediction        TEXT NOT NULL,
            phishing_prob     REAL NOT NULL,
            suspicion_score   REAL NOT NULL,
            html_fetched      INTEGER NOT NULL,
            legit_domain_known INTEGER NOT NULL,
            elapsed_sec       REAL,
            signals_json      TEXT,
            created_at        TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id    INTEGER NOT NULL,
            correct    INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_scan(result, signals):
    """Insert a completed scan into scans.db. Returns the new row id."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute(
            """INSERT INTO scans
               (url, prediction, phishing_prob, suspicion_score, html_fetched,
                legit_domain_known, elapsed_sec, signals_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                result['url'], result['prediction'], result['phishing_prob'],
                result['suspicion_score'], int(result['html_fetched']),
                int(result['legit_domain_known']), result.get('elapsed_sec'),
                json.dumps(signals), datetime.datetime.now(datetime.timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        scan_id = cur.lastrowid
        conn.close()
        return scan_id
    except Exception as e:
        print(f"[WARN] Failed to log scan: {e}")
        return None

# ── Known legitimate root domains ─────────────────────────────────────────────
LEGIT_DOMAINS = {
    'google.com','google.co.in','google.com.hk','google.co.uk','google.com.au',
    'youtube.com','gmail.com','googleapis.com','gstatic.com','google.org',
    'facebook.com','instagram.com','twitter.com','x.com','threads.net',
    'linkedin.com','microsoft.com','live.com','outlook.com','office.com',
    'apple.com','icloud.com','amazon.com','amazon.in','amazon.co.uk',
    'flipkart.com','snapdeal.com','myntra.com','meesho.com',
    'github.com','gitlab.com','stackoverflow.com','npmjs.com',
    'pinterest.com','reddit.com','tumblr.com','quora.com',
    'netflix.com','spotify.com','twitch.tv','discord.com','telegram.org',
    'paypal.com','stripe.com','razorpay.com','paytm.com','phonepe.com',
    'wikipedia.org','wikimedia.org','archive.org','britannica.com',
    'yahoo.com','bing.com','duckduckgo.com','baidu.com',
    'zoom.us','slack.com','notion.so','figma.com','canva.com',
    'shopify.com','wordpress.com','medium.com','substack.com','blogger.com',
    'nytimes.com','bbc.com','cnn.com','theguardian.com','reuters.com',
    'whatsapp.com','signal.org','skype.com',
}

def get_root_domain(domain: str) -> str:
    parts = re.sub(r':\d+$', '', domain.lower().replace('www.','')).split('.')
    return '.'.join(parts[-2:]) if len(parts) >= 2 else domain.lower()

def is_known_legit(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return get_root_domain(parsed.netloc.lower()) in LEGIT_DOMAINS

# ── HTML median fallbacks (used when page fetch fails) ────────────────────────
HTML_MEDIANS = {
    'no_of_forms': 1, 'no_of_Images': 5, 'no_of_hyperlinks': 20,
    'no_of_external_links': 5, 'no_of_internal_links': 10,
    'no_of_null_links': 2, 'no_of_selfreference_links': 3,
    'no_of_empty_links': 1, 'RatioOfEmptyLinksToExternalLinks': 0.2,
    'htmlcontent_contains_risky_extensions': 0, 'presence_of_shorten_url_html': 0,
    'hidden_elements_html': 0, 'is_script_loaded_from_external_domain': 1,
    'suspicious_inline_js': 0, 'is_footer_mismatch': 0, 'is_header_mismatch': 0,
    'is_copyright_mismatch': 0, 'is_socials_mismatch': 0, 'is_favicon_mismatch': 0,
    'has_missing_title': 0, 'title_mismatch_with_domain': 0,
    'has_inline_event_handlers': 1, 'form_action_external': 0,
}

FREE_HOSTS  = {'000webhostapp','weebly','wix','wordpress','blogspot','sites.google',
               'github.io','netlify.app','web.app','firebaseapp','square.site','glitch.me','replit.app'}
RISKY_EXT   = {'.exe','.zip','.rar','.bat','.cmd','.sh','.apk','.dmg','.msi','.iso','.jar','.scr','.pif'}
RISKY_EXT_H = {'.exe','.zip','.rar','.bat','.apk','.dmg','.msi'}
FAKE_TLDS   = {'.tk','.ml','.ga','.cf','.gq','.xyz','.top','.club','.work','.click','.link','.online','.site'}
SHORTENERS  = {'bit.ly','tinyurl.com','t.co','goo.gl','ow.ly','is.gd','buff.ly','adf.ly','tiny.cc','shorte.st'}
BRANDS      = ['paypal','amazon','google','facebook','apple','microsoft','netflix',
               'instagram','twitter','linkedin','dropbox','ebay','walmart','chase',
               'bankofamerica','wellsfargo']

PHISH_PROB_THRESHOLD = 0.65
PHISH_SUSP_THRESHOLD = 0.20


def shannon_entropy(s):
    if not s: return 0.0
    freq = Counter(s); total = len(s)
    return -sum((c/total)*math.log2(c/total) for c in freq.values())


def extract_url_features(url):
    f = {}
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower()
    path   = parsed.path.lower()
    query  = parsed.query.lower()
    dc     = re.sub(r':\d+$', '', domain)
    parts  = dc.split('.')
    tld    = '.' + parts[-1] if parts else ''

    f['brand_subdomain']        = int(any(b in '.'.join(parts[:-2]) for b in BRANDS) if len(parts)>2 else False)
    f['brand_substring_domain'] = int(any(b in dc for b in BRANDS))
    f['brand_typo_dom']         = int(any(re.sub(r'[0-9]','',dc).find(b)!=-1 and b not in dc for b in BRANDS))
    f['fake_tld']               = int(tld in FAKE_TLDS)
    f['brand_in_path_query']    = int(any(b in path+query for b in BRANDS))
    f['dots_path']              = path.count('.')

    ul = len(url); dl = len(dc)
    f['url_length']    = ul
    f['domain_length'] = dl
    f['url_more_than_mean']    = int(ul > 75)
    f['domain_more_than_mean'] = int(dl > 20)

    cons = re.sub(r'[aeiou0-9.\-_]','',dc)
    f['gibberish']       = int(bool(re.search(r'[bcdfghjklmnpqrstvwxyz]{5,}', cons)))
    f['url_contains_ip'] = int(bool(re.search(r'(\d{1,3}\.){3}\d{1,3}', domain)))
    f['presence_of_shortened_url_targeturl'] = int(any(s in dc for s in SHORTENERS))

    f['special_characters_count'] = len(re.findall(r'[^a-zA-Z0-9/:.?=&_\-#%]', url))
    for ch, key in [('@','no_of_at'),(',','no_of_comma'),('$','no_of_dollar'),(';','no_of_semicolon'),
                    (' ','no_of_space'),('&','no_of_ampersand'),('.','no_of_dots'),('=','no_of_equal'),
                    ('%','no_of_percent'),('_','no_of_underscore'),('-','no_of_hyphen'),
                    ('?','no_of_questionmark'),(':','no_of_colon')]:
        f[key] = url.count(ch)
    f['no_of_slashes_inpath'] = path.count('/')
    f['has_asterisk']     = int('*' in url)
    f['has_or']           = int('|' in url)
    f['has_embedded_url'] = int(url.lower().count('http') > 1)
    f['uses_https']       = int(parsed.scheme == 'https')

    qp = urllib.parse.parse_qs(parsed.query)
    f['num_query_params']  = len(qp)
    digits = sum(c.isdigit() for c in url)
    f['numeric_proportion'] = digits / max(ul, 1)
    f['url_entropy']        = shannon_entropy(url)
    hd = sum(c.isdigit() for c in dc)
    f['hostname_digit_ratio'] = hd / max(dl, 1)
    f['has_redirections']  = int('//' in path)
    f['has_punycode']      = int('xn--' in dc)
    f['count_www']         = url.lower().count('www')
    f['count_com']         = url.lower().count('.com')
    f['url_contains_risky_extension'] = int(any(path.endswith(e) for e in RISKY_EXT))
    f['has_explicit_port'] = int(bool(parsed.port))

    words     = [w for w in re.split(r'[^a-zA-Z0-9]+', url.replace('https','').replace('http','')) if w]
    wl        = [len(w) for w in words] if words else [0]
    f['total_words_url']         = len(words)
    f['average_length_of_words'] = float(np.mean(wl))
    f['longest_word_length']     = max(wl)
    f['shortest_word_length']    = min(wl)
    f['presence_of_free_hosting']= int(any(h in dc for h in FREE_HOSTS))
    return f


def extract_html_features(url, timeout=4):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        session = requests.Session()
        session.max_redirects = 3  # cap redirect hops — an unbounded chain (each up to `timeout`
                                    # seconds) can otherwise stack up past a host's gateway timeout
        resp = session.get(
            url,
            timeout=(3, timeout),   # (connect timeout, read timeout) — bounds worst-case per hop
                                     # to ~7s, so 1 initial request + up to 3 redirects stays
                                     # comfortably under ~28s total even in the worst case
            headers=headers, allow_redirects=True, verify=False,
        )
        soup = BeautifulSoup(resp.text, 'html.parser')
        redirect_chain = [r.url for r in resp.history] + [resp.url]
    except Exception:
        return HTML_MEDIANS.copy(), False, [url]

    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower()
    f      = {}

    f['no_of_forms']  = len(soup.find_all('form'))
    f['no_of_Images'] = len(soup.find_all('img'))
    all_links = soup.find_all('a', href=True)
    hrefs     = [a['href'] for a in all_links]
    ext  = [h for h in hrefs if h.startswith('http') and domain not in h]
    intr = [h for h in hrefs if domain in h or (not h.startswith('http') and not h.startswith('#') and h)]
    null = [h for h in hrefs if h in ('#','','javascript:void(0)','javascript:;')]
    self_r = [h for h in hrefs if h.startswith('#')]
    empty  = [h for h in hrefs if h.strip()=='']

    f['no_of_hyperlinks']           = len(all_links)
    f['no_of_external_links']       = len(ext)
    f['no_of_internal_links']       = len(intr)
    f['no_of_null_links']           = len(null)
    f['no_of_selfreference_links']  = len(self_r)
    f['no_of_empty_links']          = len(empty)
    f['RatioOfEmptyLinksToExternalLinks'] = len(empty)/max(len(ext),1)

    lt = ' '.join(hrefs).lower()
    f['htmlcontent_contains_risky_extensions'] = int(any(e in lt for e in RISKY_EXT_H))
    f['presence_of_shorten_url_html']          = int(any(s in lt for s in SHORTENERS))

    hidden = soup.find_all(style=re.compile(r'(display\s*:\s*none|visibility\s*:\s*hidden)',re.I))
    f['hidden_elements_html'] = len(hidden)

    scripts = soup.find_all('script', src=True)
    ext_sc  = [s for s in scripts if s.get('src','').startswith('http') and domain not in s.get('src','')]
    f['is_script_loaded_from_external_domain'] = int(len(ext_sc)>0)

    inline_txt = ' '.join(s.get_text() for s in soup.find_all('script', src=False)).lower()
    sus_js     = ['eval(','document.write','unescape(','fromcharcode','settimeout(','setinterval(','window.location']
    f['suspicious_inline_js'] = int(any(p in inline_txt for p in sus_js))

    foot = soup.find('footer')
    f['is_footer_mismatch'] = int(bool(foot) and domain.split('.')[0] not in foot.get_text().lower())
    head = soup.find('header')
    f['is_header_mismatch'] = int(bool(head) and domain.split('.')[0] not in head.get_text().lower())
    cp = re.search(r'©|copyright', resp.text, re.I)
    f['is_copyright_mismatch'] = int(bool(cp) and domain.split('.')[0] not in (cp.group(0) if cp else ''))

    social_doms = ['facebook.com','twitter.com','instagram.com','linkedin.com','youtube.com']
    soc_links   = [a['href'] for a in all_links if any(s in a.get('href','') for s in social_doms)]
    f['is_socials_mismatch'] = int(len(soc_links)>0 and not any(domain.split('.')[0] in s for s in soc_links))

    fav = soup.find('link', rel=re.compile(r'icon',re.I))
    fh  = fav.get('href','') if fav else ''
    f['is_favicon_mismatch'] = int(fh.startswith('http') and domain not in fh)

    title = soup.find('title')
    tt    = title.get_text().lower() if title else ''
    f['has_missing_title']          = int(not bool(tt.strip()))
    f['title_mismatch_with_domain'] = int(bool(tt) and domain.split('.')[0] not in tt)

    evts = ['onclick','onmouseover','onload','onsubmit','onerror']
    f['has_inline_event_handlers'] = int(any(soup.find_all(attrs={e:True}) for e in evts))

    forms    = soup.find_all('form', action=True)
    ext_frms = [fm for fm in forms if fm['action'].startswith('http') and domain not in fm['action']]
    f['form_action_external'] = int(len(ext_frms)>0)
    return f, True, redirect_chain


def compute_suspicion_score_single(features, estats):
    def norm(v, mn, mx): return max(0.0, min(1.0, (v-mn)/(mx-mn+1e-9)))
    s1 = norm(features.get('url_entropy',0), estats.get('url_entropy_min',0), estats.get('url_entropy_max',1))
    spc = sum(features.get(c,0) for c in ['no_of_at','no_of_percent','no_of_hyphen','no_of_underscore','no_of_semicolon','no_of_dollar'])
    s2  = norm(spc/max(features.get('url_length',1),1), estats.get('spc_density_min',0), estats.get('spc_density_max',1))
    s3  = norm(features.get('domain_length',0), estats.get('domain_length_min',0), estats.get('domain_length_max',1))
    ext = features.get('no_of_external_links',0); intr = features.get('no_of_internal_links',0)
    s4  = norm(ext/(ext+intr+1), estats.get('ext_ratio_min',0), estats.get('ext_ratio_max',1))
    return round(float(np.mean([s1,s2,s3,s4])), 4)


def compute_shap_top_features(X_sc, top_n=5):
    """Returns the top-N features pushing this prediction, or [] if SHAP unavailable."""
    explainer = get_shap_explainer()
    if explainer is None:
        return []
    try:
        sv = explainer.shap_values(X_sc)
        sv = np.array(sv).reshape(-1)  # (n_features,) — contribution toward "phishing"
        order = np.argsort(-np.abs(sv))[:top_n]
        out = []
        for i in order:
            feat_name = ALL_FEATURES[i]
            val = float(sv[i])
            out.append({
                'feature': feat_name,
                'impact': round(val, 4),
                'direction': 'increases_risk' if val > 0 else 'decreases_risk',
            })
        return out
    except Exception as e:
        print(f"[WARN] SHAP explanation failed: {e}")
        return []


def check_safe_browsing(url, timeout=4):
    """Optional cross-check against Google Safe Browsing v4. Returns None if no API key
    is configured (GOOGLE_SAFE_BROWSING_API_KEY env var) or the lookup fails/times out —
    never blocks or breaks the main prediction."""
    if not GOOGLE_SAFE_BROWSING_API_KEY:
        return None
    try:
        endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={GOOGLE_SAFE_BROWSING_API_KEY}"
        body = {
            "client": {"clientId": "attack-on-phishing", "clientVersion": "1.0"},
            "threatInfo": {
                "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [{"url": url}],
            },
        }
        r = requests.post(endpoint, json=body, timeout=timeout)
        r.raise_for_status()
        matches = r.json().get("matches", [])
        return {"flagged": len(matches) > 0, "match_count": len(matches)}
    except Exception as e:
        print(f"[INFO] Safe Browsing lookup skipped: {e}")
        return {"error": "lookup_unavailable"}


def run_single_prediction(raw_url, check_safe_browse=True):
    """Runs the full pipeline for one URL. Same math/thresholds as the original
    /predict route — this is a refactor for reuse by /predict and /predict/batch,
    not a behavior change. Returns (result_dict, signals_dict_for_logging)."""
    raw_url = raw_url.strip()
    if not raw_url.startswith(('http://', 'https://')):
        raw_url = 'https://' + raw_url

    t0 = time.time()
    legit_override                       = is_known_legit(raw_url)
    url_feats                            = extract_url_features(raw_url)
    html_feats, html_fetched, redirect_chain = extract_html_features(raw_url)
    all_feats                            = {**url_feats, **html_feats}
    suspicion_score                      = compute_suspicion_score_single(all_feats, ESTATS)
    all_feats['suspicion_score']         = suspicion_score

    row   = {f: all_feats.get(f, 0) for f in ALL_FEATURES}
    X     = pd.DataFrame([row], columns=ALL_FEATURES)
    X_imp = pd.DataFrame(imputer.transform(X),   columns=ALL_FEATURES)
    X_sc  = pd.DataFrame(scaler.transform(X_imp), columns=ALL_FEATURES)
    prob  = float(model.predict_proba(X_sc)[0, 1])

    # Dual-threshold decision (unchanged from v2)
    _host = urllib.parse.urlparse(raw_url).netloc.lower()
    _tld  = '.' + _host.split('.')[-1] if '.' in _host else ''
    _is_ip        = bool(re.search(r'^(\d{1,3}\.){3}\d{1,3}(:\d+)?$', re.sub(r':\d+$', '', _host)))
    _is_shortener = _host in SHORTENERS or any(_host == s or _host.endswith('.' + s) for s in SHORTENERS)
    _is_fake_tld  = _tld in FAKE_TLDS

    if _is_ip or _is_shortener or _is_fake_tld:
        pred = 1
    elif legit_override:
        pred = int(prob > 0.85 and suspicion_score > 0.45)
    else:
        pred = int(prob >= PHISH_PROB_THRESHOLD and suspicion_score >= PHISH_SUSP_THRESHOLD)

    elapsed = round(time.time() - t0, 2)

    signals = {
        'url_length':               url_feats.get('url_length'),
        'uses_https':               url_feats.get('uses_https'),
        'url_entropy':              round(url_feats.get('url_entropy', 0), 4),
        'has_ip_in_url':            url_feats.get('url_contains_ip'),
        'no_of_external_links':     html_feats.get('no_of_external_links'),
        'suspicious_inline_js':     html_feats.get('suspicious_inline_js'),
        'presence_of_free_hosting': url_feats.get('presence_of_free_hosting'),
        'form_action_external':     html_feats.get('form_action_external'),
        'fake_tld':                 url_feats.get('fake_tld'),
        'has_punycode':             url_feats.get('has_punycode'),
        'no_of_forms':              html_feats.get('no_of_forms'),
        'hidden_elements':          html_feats.get('hidden_elements_html'),
    }

    result = {
        'url':                raw_url,
        'prediction':         'phishing' if pred == 1 else 'legitimate',
        'phishing_prob':      round(prob, 4),
        'suspicion_score':    suspicion_score,
        'confidence_pct':     round(prob * 100 if pred == 1 else (1 - prob) * 100, 1),
        'elapsed_sec':        elapsed,
        'html_fetched':       html_fetched,
        'legit_domain_known': legit_override,
        'features':           signals,
        'redirect_chain':     redirect_chain,
        'shap_top_features':  compute_shap_top_features(X_sc),
        'safe_browsing':      check_safe_browsing(raw_url) if check_safe_browse else None,
    }
    return result, signals


@app.route('/', methods=['GET'])
def serve_frontend():
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'loaded'})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'Provide {"url": "..."}'}), 400

    result, signals = run_single_prediction(data['url'])
    result['scan_id'] = log_scan(result, signals)
    return jsonify(result)


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Accepts either a JSON body {"urls": ["...", "..."]} or a multipart file
    upload (field name 'file') containing a CSV with a 'url' column (or the
    URLs in the first column if there's no header named 'url')."""
    urls = []

    if 'file' in request.files:
        f = request.files['file']
        try:
            text = f.read().decode('utf-8', errors='ignore')
            reader = csv.reader(io.StringIO(text))
            rows = list(reader)
            if not rows:
                return jsonify({'error': 'CSV file is empty'}), 400
            header = [c.strip().lower() for c in rows[0]]
            if 'url' in header:
                col = header.index('url')
                urls = [r[col].strip() for r in rows[1:] if len(r) > col and r[col].strip()]
            else:
                # No recognizable header — treat every non-empty first cell as a URL
                urls = [r[0].strip() for r in rows if r and r[0].strip()]
        except Exception as e:
            return jsonify({'error': f'Could not parse CSV: {e}'}), 400
    else:
        data = request.get_json(silent=True) or {}
        urls = [u.strip() for u in data.get('urls', []) if isinstance(u, str) and u.strip()]

    if not urls:
        return jsonify({'error': 'No URLs found. Provide {"urls": [...]} or a CSV file with a "url" column.'}), 400

    MAX_BATCH = 100
    truncated = len(urls) > MAX_BATCH
    urls = urls[:MAX_BATCH]

    results = []
    for u in urls:
        try:
            # Skip the Safe Browsing round-trip per-URL in batch mode to keep it snappy.
            result, signals = run_single_prediction(u, check_safe_browse=False)
            result['scan_id'] = log_scan(result, signals)
            results.append(result)
        except Exception as e:
            results.append({'url': u, 'error': str(e)})

    return jsonify({'count': len(results), 'truncated': truncated, 'results': results})


@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json(silent=True) or {}
    scan_id = data.get('scan_id')
    correct = data.get('correct')
    if scan_id is None or correct is None:
        return jsonify({'error': 'Provide {"scan_id": <int>, "correct": <bool>}'}), 400
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO feedback (scan_id, correct, created_at) VALUES (?,?,?)",
            (int(scan_id), int(bool(correct)), datetime.datetime.now(datetime.timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dashboard/stats', methods=['GET'])
def dashboard_stats():
    """Aggregate stats for the analytics dashboard: totals, ratio, trend, top signals."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        total = conn.execute("SELECT COUNT(*) AS c FROM scans").fetchone()['c']
        phishing_count = conn.execute(
            "SELECT COUNT(*) AS c FROM scans WHERE prediction='phishing'"
        ).fetchone()['c']
        legit_count = total - phishing_count
        avg_susp = conn.execute("SELECT AVG(suspicion_score) AS a FROM scans").fetchone()['a'] or 0

        # Last 14 days trend, grouped by date
        trend_rows = conn.execute("""
            SELECT substr(created_at,1,10) AS day,
                   COUNT(*) AS count,
                   AVG(suspicion_score) AS avg_susp,
                   SUM(CASE WHEN prediction='phishing' THEN 1 ELSE 0 END) AS phishing_count
            FROM scans
            GROUP BY day
            ORDER BY day DESC
            LIMIT 14
        """).fetchall()
        trend = [
            {
                'date': r['day'],
                'count': r['count'],
                'avg_suspicion': round(r['avg_susp'] or 0, 4),
                'phishing_count': r['phishing_count'],
            }
            for r in reversed(trend_rows)
        ]

        # Top offending signals among URLs flagged as phishing
        phishing_rows = conn.execute(
            "SELECT signals_json FROM scans WHERE prediction='phishing' AND signals_json IS NOT NULL"
        ).fetchall()
        conn.close()

        flag_checks = {
            'fake_tld':                 lambda s: s.get('fake_tld') == 1,
            'has_ip_in_url':            lambda s: s.get('has_ip_in_url') == 1,
            'presence_of_free_hosting': lambda s: s.get('presence_of_free_hosting') == 1,
            'suspicious_inline_js':     lambda s: s.get('suspicious_inline_js') == 1,
            'form_action_external':     lambda s: s.get('form_action_external') == 1,
            'has_punycode':             lambda s: s.get('has_punycode') == 1,
            'no_https':                 lambda s: s.get('uses_https') == 0,
            'long_url':                 lambda s: (s.get('url_length') or 0) > 75,
            'high_entropy':             lambda s: (s.get('url_entropy') or 0) > 4.5,
        }
        n_phish = len(phishing_rows)
        top_signals = []
        if n_phish > 0:
            for name, check in flag_checks.items():
                count = 0
                for row in phishing_rows:
                    try:
                        s = json.loads(row['signals_json'])
                        if check(s):
                            count += 1
                    except Exception:
                        continue
                if count > 0:
                    top_signals.append({
                        'signal': name,
                        'count': count,
                        'pct_of_phishing': round(100 * count / n_phish, 1),
                    })
            top_signals.sort(key=lambda x: -x['pct_of_phishing'])

        return jsonify({
            'total_scans':      total,
            'phishing_count':   phishing_count,
            'legit_count':      legit_count,
            'avg_suspicion':    round(avg_susp, 4),
            'trend':            trend,
            'top_signals':      top_signals[:8],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # reloader_type='stat' avoids the watchdog reloader treating our own
    # scans.db writes as a source-code change and restarting the server.
    app.run(debug=True, port=5000, reloader_type='stat')
