# ML / Data Analysis Assessment (Beginner Friendly)

> "Can a handful of simple rules and a few lines of Python separate real people from generic inboxes? I built this project to prove the answer is yes—and to show the stepping stones from raw data to explainable classification." 

I built a small end‑to‑end mini project that cleans an email contact dataset, explores it (EDA), classifies emails into **generic** vs **non-generic (personal)**, and (optionally) trains a very light ML model + runs unit tests. I kept everything intentionally simple so a reviewer (or future me) can follow the logic at a glance.

---
## � How I Run Everything (Terminal Quick Start)
Below are the exact commands I use on Windows PowerShell. Replace `<REPO_URL>` when cloning.

```powershell
# 1. Clone
https://github.com/NiranjanGhising/ML-and-Data_Analayst_Project.git
cd email_classifier_project

# 2. (Optional) Create & activate virtual environment
python -m venv .venv
 .\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install pandas scikit-learn numpy openpyxl pytest

# 4. Run rule + (optional) ML classification script (Task 3)
python main.py

# 5. Run minimal tests (email validation + rules)
python -m pytest -q

# 6. (Optional) Open EDA notebook (if Jupyter installed)
# jupyter notebook  # then open Travel_EDA.ipynb
```

Expected highlights after `python main.py`:
- Baseline vs refined rule counts
- Logistic Regression metrics (accuracy / precision / recall / F1)
- Confusion matrix
- Best threshold + top feature coefficients

If something breaks: re-run environment activation and reinstall packages.

---
## �🎯 What I Practiced (and You Can Learn From Reading)
| Topic | What I implemented |
|-------|-------------------|
| Data Cleaning | Removing duplicates, handling missing values, validating emails |
| Normalization | Consistent column names, simple string normalization |
| Rule-Based Logic | Baseline vs refined heuristic classification |
| Exploratory Data Analysis | Counts, top companies, missing summary, labeled charts |
| Simple ML (Optional) | Logistic Regression trained on rule labels (weak supervision) |
| Evaluation | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Testing | Minimal pytest tests for email rules |
| (Optional) DB | Export cleaned & labeled data to PostgreSQL |

---
## 🗂 Project Structure (Key Files)
```
Project--
  travel_emails.xlsx        # Source dataset (sample) - required
  email_rules.py            # Simple baseline + refined classification logic
  main.py                   # Task 3 script: rules + optional ML + metrics
  Travel_EDA.ipynb          # Notebook for Task 1 + Task 2 (clean + EDA + rule labels)
  tests/
    test_email_rules.py     # Minimal concept-focused tests (pytest)
  README.md                 # (You are here)
```

---
## ✅ Task Mapping (Prompt -> Implementation)
| Assessment Task | Where Implemented |
|-----------------|-------------------|
| Load & clean, keep required columns | Notebook + logic summary (Travel_EDA.ipynb) |
| Email validation (regex) | `email_rules.py` (`validate_email`) |
| Generic vs non-generic labeling | Baseline + refined (`email_rules.py`, notebook, `main.py`) |
| Stats (totals, counts, missing) | Notebook cells (EDA) |
| Visualizations (>=2) | Notebook (bar chart + pie chart + top companies chart) |
| Classifier (rule-based) | `email_rules.py` + `main.py` |
| Optional ML model | `main.py` (Logistic Regression) |
| Metrics (accuracy, precision, recall, confusion matrix) | Printed in `main.py` run |
| Tests for validation logic | `tests/test_email_rules.py` |
| (Optional) Database export | (Logic outlined; add script if desired) |
| Explanations / comments | In notebook + code docstrings/comments |

---
## 🧠 Core Concepts (Plain English)
**Generic email**: A shared or functional mailbox (info@, support@, sales@) that doesn’t belong to a single person.
**Non-generic email**: A personal / individual contact (alice@company.com) useful for direct outreach.

### Baseline Rule (Start Simple)
I began with: if the local part starts with one of `info`, `support`, `admin`, `hello` → generic. One line. Easy to explain. High precision. Low recall.

### Refined Rule (Add Smart but Still Simple Signals)
I layered more logic while keeping readability:
1. Expanded functional token list (sales, hr, booking, team...).  
2. Catch‑all pattern: local == domain root (brand@brand.com).  
3. Company similarity: local equals / contains / is contained in company name.  
4. Personal guard: if both first & last name fragments appear → force non-generic.  
Result: higher recall without throwing away clarity.

### Optional ML Layer (In `main.py`)
I trained a tiny Logistic Regression on the refined rule labels (weak supervision). Why? To:
* See which engineered features the model leans on.
* Try threshold tuning for a better precision / recall balance.
* Demonstrate a path from rules → model without needing manual labels.

---
## 🛠 Prerequisites
- Python 3.10+ recommended
- pip

(Optional) create a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies (add `pytest` if not already in requirements):
```powershell
pip install pandas scikit-learn numpy openpyxl pytest
```

---
## 🚀 Quick Start
Clone the repository:
```powershell
git clone <YOUR_REPO_URL> internship_email_project
cd internship_email_project
```
Ensure `travel_emails.xlsx` is in the root folder (or replace with your sheet named appropriately).

Run the EDA notebook (open in VS Code / Jupyter, execute cells in order) to see cleaning + derived stats:
1. Preview data
2. Derive generic/non-generic label (rule logic in notebook cell)
3. Stats + charts

Run the classification + metrics script:
```powershell
python main.py
```
Outputs include:
- Baseline vs refined rule comparison
- Logistic Regression metrics (accuracy, precision, recall, F1)
- Confusion matrix
- Threshold tuning result & top feature weights

Run tests:
```powershell
python -m pytest -q
```
All tests should pass (they are intentionally minimal and easy to explain).

---
## 🧪 Interpreting The Output
| Output | Meaning |
|--------|---------|
| Baseline generic count | How many emails the smallest rule catches |
| Refined generic count | How many after smarter heuristics |
| Precision | Of predicted generics, how many were (heuristic) true |
| Recall | How many heuristic generics we successfully found |
| F1 | Balance between precision & recall |
| Threshold scan | Best probability cutoff for F1 |
| Top features | Which signals most influence generic classification |

---
## 📝 How I Explain My Approach (Script Version)
I started with a very simple prefix rule for generic emails. To improve coverage, I added more functional words and simple similarity checks between the email local part, the domain root, and the company name. I protected precision by checking if both a first and last name were present inside the email. Then I optionally trained a logistic regression model using those refined labels as weak supervision to inspect which features matter most (like generic prefixes or digits). I evaluated the model with accuracy, precision, recall, F1, and a confusion matrix, and tuned the threshold for best F1.

---
## 🗃 (Optional) PostgreSQL / SQLite Export
I can extend this by writing the labeled DataFrame to a database table for downstream analytics (e.g., using SQLAlchemy). If I add a script:
```python
df.to_sql("emails_categorized", engine, if_exists="replace", index=False)
```
I would add indexes for faster lookups on email or company.


---
## ❓ Troubleshooting
| Problem | Fix |
|---------|-----|
| ModuleNotFoundError (pandas/sklearn) | Re-run `pip install ...` inside correct environment |
| Tests fail due to path | Run from project root: `python -m pytest -q` |
| Excel read error | Ensure file name matches `travel_emails.xlsx` or adjust code |
| Non-UTF8 errors reading file | Already handled by fallback in `main.py` |



