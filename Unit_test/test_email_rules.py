"""
tests/test_email_rules.py (Concept-Focused Minimal Tests)

"""

from email_rules import validate_email, baseline, refined, make_row

# 1. Validation -----------------------------------------------------------
def test_validate_good():  # clearly valid formats
    assert validate_email('info@example.com')
    assert validate_email('john.doe+tag@sub.domain.co')

def test_validate_bad():  # structurally broken patterns
    assert not validate_email('noatsymbol')          # missing '@'
    assert not validate_email('user@domain')         # missing final dot section
    assert not validate_email('user@domain.c')       # TLD too short

# 2. Baseline classification --------------------------------------------
def test_baseline_generic():
    assert baseline('info@company.com') == 'generic'  # prefix hit

def test_baseline_non_generic():
    assert baseline('alice@company.com') == 'non-generic'  # personal name

# 3. Refined rule adds more signals -------------------------------------
def test_refined_functional_token():  # expanded functional word
    row = make_row('sales@brand.io', '', '', 'Brand')
    assert refined(row) == 'generic'

def test_refined_domain_root_match():  # brand@brand.com style catch all
    row = make_row('mybrand@mybrand.com', 'Ann', 'Lee', 'MyBrand')
    assert refined(row) == 'generic'

def test_refined_personal_guard():  # first + last name should stay non-generic
    row = make_row('john.smith@mybrand.com', 'John', 'Smith', 'MyBrand')
    assert refined(row) == 'non-generic'
