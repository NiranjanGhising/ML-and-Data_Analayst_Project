"""
TASK 3: Generic vs Non-Generic Email Classification
===================================================

"""

import pandas as pd, re, sys, os, numpy as np
from collections import Counter
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

SOURCE_FILE = "travel_emails.xlsx"  # default expected name

# ============================
# DATA LOADING & VALIDATION
# ============================
# Enhanced: search common locations so script works whether run from project root
# or inside the 'Task 3' folder.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))  # parent directory of Task 3

candidate_paths = [
    # 1. Current working directory
    os.path.join(os.getcwd(), SOURCE_FILE),
    # 2. Script directory (Task 3)
    os.path.join(script_dir, SOURCE_FILE),
    # 3. Project root
    os.path.join(project_root, SOURCE_FILE),
    # 4. Task 1 subfolder (where the dataset actually resides)
    os.path.join(project_root, 'Task 1', SOURCE_FILE),
    # 5. One level up from root just in case
    os.path.abspath(os.path.join(project_root, os.pardir, SOURCE_FILE)),
]

data_path = None
for p in candidate_paths:
    if os.path.exists(p):
        data_path = p
        break

if not data_path:
    print("ERROR: Could not locate the dataset. Searched these paths:")
    for p in candidate_paths:
        print(" -", p)
    sys.exit(1)

print(f"Using dataset: {data_path}")
df = pd.read_excel(data_path)

#if columns missing, report and exit
required_cols = {"first_name","last_name","company_name","email"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    print(f"ERROR: Missing required columns: {missing_cols}")
    print("Columns present:", list(df.columns))
    sys.exit(1)

# ============================
# CORE RESOURCES & NORMALIZATION UTILITIES
# ============================
BASE_GENERIC_PREFIXES = {
    'info','support','admin','hello','sales','booking','reservations','reservation','enquiry','inquiry',
    'contact','marketing','press','careers','career','hr','office','mail','team','help','service','services'
}

def norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

# Personal-like regex patterns:
#  - first.last variants
#  - fLastname (initial + surname) patterns
#  - name followed by digits
PERSONAL_PATTERNS = [
    re.compile(r'^[a-z]+[._-][a-z]+$'),          # first.last / first_last
    re.compile(r'^[a-z]{1}[a-z]+[0-9]*$'),       # fLastname / fLastname123
    re.compile(r'^[a-z]+[0-9]+$'),               # name123
]

def looks_personal(local_norm: str, first_norm: str, last_norm: str) -> bool:
    if first_norm and last_norm and first_norm in local_norm and last_norm in local_norm:
        return True
    for pat in PERSONAL_PATTERNS:
        if pat.match(local_norm):
            return True
    return False

# ============================
# AUTO-MINING FUNCTIONAL TOKENS (Heuristic Step 5)
# ============================
# Identify frequently occurring local parts likely representing role or group mailboxes.
# Frequency threshold chosen (>=5) to avoid overfitting to one-offs; adjustable.
locals_series = df['email'].astype(str).str.lower().str.split('@').str[0].fillna('')
first_norms = df['first_name'].fillna('').map(norm)
last_norms  = df['last_name'].fillna('').map(norm)

counts = Counter(locals_series)
auto_candidates = {
    lp for lp,cnt in counts.items()
    if cnt >= 5 and 3 <= len(lp) <= 15 and lp.isalpha()
}
personal_frags = set(first_norms.unique()) | set(last_norms.unique())
auto_candidates = {lp for lp in auto_candidates if not any(f and f in lp for f in personal_frags)}

GENERIC_SET = set(BASE_GENERIC_PREFIXES) | auto_candidates

print(f"Auto-mined functional tokens added: {sorted(auto_candidates)}")

# ============================
# RULE IMPLEMENTATIONS (Baseline vs Refined)
# ============================
# baseline(): Minimal spec requirement; small high-precision prefix list.
# refined(): Expanded multi-signal heuristic that aims for higher recall while
#            guarding against false positives via personal name detection.
def baseline(email: str) -> str:
    """Baseline: original short prefix rule only."""
    if '@' not in email:
        return 'non-generic'
    lp = email.split('@')[0].lower()
    return 'generic' if any(lp.startswith(p) for p in ('info','support','admin','hello')) else 'non-generic'

def refined(row) -> str:
    email = str(row.email).strip().lower()
    if '@' not in email:
        return 'non-generic'
    local, domain_full = email.split('@',1)
    domain_root = domain_full.split('.')[0]
    ln_local = norm(local)
    dn_root = norm(domain_root)
    company_norm = norm(row.company_name)
    first_norm = norm(row.first_name)
    last_norm  = norm(row.last_name)
    personal_both = (first_norm and last_norm and first_norm in ln_local and last_norm in ln_local)

    # A: functional token prefix or exact token
    if any(local.startswith(p) for p in GENERIC_SET) or ln_local in GENERIC_SET:
        if not personal_both:
            return 'generic'

    # B: local equals domain root
    if dn_root and ln_local == dn_root and not personal_both:
        return 'generic'

    # C: local matches company (subset/equality)
    if company_norm and (ln_local == company_norm or company_norm in ln_local or ln_local in company_norm) and not personal_both:
        return 'generic'

    return 'non-generic'

# ============================
# APPLY RULES & COMPARE COVERAGE
# ============================
baseline_label = df['email'].astype(str).apply(baseline)
refined_label = df.apply(refined, axis=1)

# Evaluation baseline vs refined
ct_rules = pd.crosstab(baseline_label, refined_label, rownames=['baseline'], colnames=['refined'])
extra_generics = (refined_label == 'generic').sum() - (baseline_label == 'generic').sum()

print("\n=== Rule Comparison (Baseline vs Refined) ===")
print(f"Total records: {len(df)}")
print("Confusion matrix (rows=baseline, cols=refined):\n", ct_rules)
print(f"Baseline generic count: {(baseline_label=='generic').sum()}  Refined generic count: {(refined_label=='generic').sum()}  (+{extra_generics})")
print(f"Baseline recall vs refined (proxy truth) = {recall_score(refined_label, baseline_label, pos_label='generic', zero_division=0):.3f}")

# ============================
# WEAK-SUPERVISED ML TRAINING (Step 6)
# ============================
# Use refined rule labels as 'teacher'. Logistic Regression chosen for:
#  - Interpretability (coefficients)
#  - Speed and simplicity for small feature set
#  - Handles sparse one-hot features well

print("\n=== Weak-Supervised Logistic Regression (target = refined rule) ===")
emails = df['email'].astype(str).str.lower()
local_part = emails.str.split('@').str[0].fillna('')
domain_part = emails.str.split('@').str[1].fillna('')
domain_root = domain_part.str.split('.').str[0]
top_level = domain_part.str.split('.').str[-1]

X = pd.DataFrame({
    'local_len': local_part.str.len(),
    'has_digit': local_part.str.contains(r'\d').astype(int),
    'num_digits': local_part.str.count(r'\d'),
    'has_dot_local': local_part.str.contains(r'\.').astype(int),
    'has_underscore': local_part.str.contains(r'_').astype(int),
    'domain_root': domain_root,
    'tld': top_level,
    'company_domain_match': [
        int(norm(c) and norm(d) and (norm(c)==norm(d) or norm(c) in norm(d) or norm(d) in norm(c)))
        for c,d in zip(df['company_name'], domain_root)
    ],
    'starts_generic_prefix': [int(any(lp.startswith(p) for p in BASE_GENERIC_PREFIXES)) for lp in local_part],
    'auto_mined_hit': [int(lp in auto_candidates) for lp in local_part],
})

y = refined_label

cat_cols = ['domain_root','tld']
num_cols = [c for c in X.columns if c not in cat_cols]

pre = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=5), cat_cols),
    ('num', 'passthrough', num_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

pipe = Pipeline([
    ('prep', pre),
    ('clf', LogisticRegression(max_iter=600, class_weight='balanced'))
])

pipe.fit(X_train, y_train)
probs_test = pipe.predict_proba(X_test)[:, pipe.classes_.tolist().index('generic')]
y_pred_default = pipe.predict(X_test)

acc = accuracy_score(y_test, y_pred_default)
prec = precision_score(y_test, y_pred_default, pos_label='generic', zero_division=0)
rec = recall_score(y_test, y_pred_default, pos_label='generic', zero_division=0)
f1 = f1_score(y_test, y_pred_default, pos_label='generic', zero_division=0)
cm = confusion_matrix(y_test, y_pred_default, labels=['generic','non-generic'])

print(f"Default threshold metrics -> Acc: {acc:.3f} Prec: {prec:.3f} Rec: {rec:.3f} F1: {f1:.3f}")
print("Confusion matrix (rows=true, cols=pred):\n", cm)

# Threshold tuning for max F1
best_t, best_f1 = 0.5, -1
for t in [i/100 for i in range(20,81)]:  # 0.20 .. 0.80
    preds_t = np.where(probs_test >= t, 'generic','non-generic')
    f1_t = f1_score(y_test, preds_t, pos_label='generic', zero_division=0)
    if f1_t > best_f1:
        best_f1, best_t = f1_t, t

preds_best = np.where(probs_test >= best_t, 'generic','non-generic')
acc_b = accuracy_score(y_test, preds_best)
prec_b = precision_score(y_test, preds_best, pos_label='generic', zero_division=0)
rec_b = recall_score(y_test, preds_best, pos_label='generic', zero_division=0)
print(f"Best threshold {best_t:.2f} -> Acc: {acc_b:.3f} Prec: {prec_b:.3f} Rec: {rec_b:.3f} F1: {best_f1:.3f}")

# Top feature coefficients (absolute)
ohe_features = pipe.named_steps['prep'].transformers_[0][1].get_feature_names_out(cat_cols).tolist()
feature_names = ohe_features + num_cols
coefs = pipe.named_steps['clf'].coef_[0]
coef_df = (pd.DataFrame({'feature': feature_names, 'coef': coefs})
           .assign(abs_coef=lambda d: d['coef'].abs())
           .sort_values('abs_coef', ascending=False)
           .head(12))
print("Top contributing features:\n", coef_df)

print("\nDone.")