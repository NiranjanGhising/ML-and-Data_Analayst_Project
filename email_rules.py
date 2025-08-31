"""email_rules.py (Recreated)
Simple, explainable generic vs non-generic email classification for tests.
"""
import re
from types import SimpleNamespace

GENERIC_SET = {
	'info','support','admin','hello','sales','booking','reservations',
	'contact','hr','team','press','marketing','office'
}

EMAIL_REGEX = re.compile(r'^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$', re.I)

def validate_email(email: str) -> bool:
	return bool(EMAIL_REGEX.match(str(email).strip()))

def norm(s: str) -> str:
	return re.sub(r'[^a-z0-9]', '', str(s).lower())

def baseline(email: str) -> str:
	if not validate_email(email):
		return 'non-generic'
	local = email.split('@')[0].lower()
	return 'generic' if any(local.startswith(p) for p in ('info','support','admin','hello')) else 'non-generic'

def refined(row) -> str:
	email = str(row.email).strip().lower()
	if not validate_email(email):
		return 'non-generic'
	local, domain_full = email.split('@',1)
	domain_root = domain_full.split('.')[0]
	ln_local = norm(local)
	dn_root = norm(domain_root)
	company_norm = norm(getattr(row,'company_name',''))
	first_norm = norm(getattr(row,'first_name',''))
	last_norm  = norm(getattr(row,'last_name',''))
	personal = first_norm and last_norm and first_norm in ln_local and last_norm in ln_local
	if (any(local.startswith(p) for p in GENERIC_SET) or ln_local in GENERIC_SET) and not personal:
		return 'generic'
	if dn_root and ln_local == dn_root and not personal:
		return 'generic'
	if company_norm and not personal:
		if ln_local == company_norm or company_norm in ln_local or ln_local in company_norm:
			return 'generic'
	return 'non-generic'

def make_row(email, first='', last='', company=''):
	return SimpleNamespace(email=email, first_name=first, last_name=last, company_name=company)
