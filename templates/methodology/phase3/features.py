import math
import re
from collections import Counter
from urllib.parse import urlsplit

import tldextract

# Use the suffix list included in the library (without downloading anything).
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "s.id",
}
# Words that commonly appear in impersonation attempts.
SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "verification", "secure", "account", "update",
    "confirm", "password", "bank", "banco", "wallet", "billing", "invoice",
    "support", "suspend", "unlock", "pago", "clave", "token", "auth",
]
EXECUTABLE_EXT = (".exe", ".dll", ".sh", ".bat", ".ps1", ".vbs", ".hta", ".apk",
                  ".msi", ".jar", ".scr", ".zip", ".rar", ".bin")
RISKY_TLDS = {"tk", "ml", "ga", "cf", "gq", "top", "xyz", "icu", "cyou", "click",
              "work", "zip", "mov", "buzz", "rest", "monster"}

_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")

FEATURE_COLUMNS = [
    "url_length", "host_length", "path_length", "query_length",
    "n_dots_url", "n_hyphens_url", "n_underscores_url", "n_slashes_url",
    "n_digits_url", "digit_ratio_url", "n_special_chars_url",
    "n_subdomains", "host_n_hyphens", "host_n_digits", "host_digit_ratio",
    "has_ip_host", "has_port", "port_is_nonstandard",
    "path_depth", "n_query_params", "has_at_symbol", "has_double_slash_in_path",
    "has_punycode", "is_shortener", "n_suspicious_words",
    "has_executable_ext", "risky_tld", "host_entropy", "url_entropy",
    "longest_host_token", "tld_length",
]

def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())

def normalize_url(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw).strip().strip('"').strip("'")
    s = s.replace("hxxp", "http").replace("[.]", ".")
    if not s:
        return ""
    if not _SCHEME.match(s):
        s = "http://" + s
    try:
        p = urlsplit(s)
    except ValueError:
        return ""
    host = (p.hostname or "").rstrip(".").lower()
    if not host:
        return ""
    marca = "u@" if "@" in p.netloc else ""
    netloc = marca + host + (f":{p.port}" if p.port else "")
    path = p.path or ""
    query = f"?{p.query}" if p.query else ""
    return f"http://{netloc}{path}{query}"

def registered_domain(normalized_url: str) -> str:
    host = urlsplit(normalized_url).hostname or ""
    if _IPV4.match(host):
        return host
    ext = _EXTRACT(host)
    return ".".join(x for x in (ext.domain, ext.suffix) if x) or host

def extract_features(normalized_url: str) -> dict:
    p = urlsplit(normalized_url)
    host = p.hostname or ""
    path, query = p.path or "", p.query or ""
    url_sin_esquema = ("u@" if "@" in (p.netloc or "") else "") + host + (f":{p.port}" if p.port else "") + path + (f"?{query}" if query else "")
    ext = _EXTRACT(host)
    es_ip = bool(_IPV4.match(host))
    n_sub = 0 if es_ip else len([s for s in ext.subdomain.split(".") if s])
    n_dig_url = sum(c.isdigit() for c in url_sin_esquema)
    n_dig_host = sum(c.isdigit() for c in host)
    low = url_sin_esquema.lower()
    port = p.port

    return {
        "url_length": len(url_sin_esquema),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "n_dots_url": url_sin_esquema.count("."),
        "n_hyphens_url": url_sin_esquema.count("-"),
        "n_underscores_url": url_sin_esquema.count("_"),
        "n_slashes_url": url_sin_esquema.count("/"),
        "n_digits_url": n_dig_url,
        "digit_ratio_url": n_dig_url / max(len(url_sin_esquema), 1),
        "n_special_chars_url": len(re.findall(r"[^A-Za-z0-9./:\-_]", url_sin_esquema)),
        "n_subdomains": n_sub,
        "host_n_hyphens": host.count("-"),
        "host_n_digits": n_dig_host,
        "host_digit_ratio": n_dig_host / max(len(host), 1),
        "has_ip_host": int(es_ip),
        "has_port": int(port is not None),
        "port_is_nonstandard": int(port is not None and port not in (80, 443)),
        "path_depth": len([s for s in path.split("/") if s]),
        "n_query_params": len([q for q in query.split("&") if q]),
        "has_at_symbol": int("@" in url_sin_esquema),
        "has_double_slash_in_path": int("//" in path),
        "has_punycode": int("xn--" in host),
        "is_shortener": int(host in SHORTENERS),
        "n_suspicious_words": sum(w in low for w in SUSPICIOUS_WORDS),
        "has_executable_ext": int(path.lower().endswith(EXECUTABLE_EXT)),
        "risky_tld": int(ext.suffix.split(".")[-1] in RISKY_TLDS),
        "host_entropy": _entropy(host),
        "url_entropy": _entropy(url_sin_esquema),
        "longest_host_token": max((len(t) for t in re.split(r"[.\-]", host)), default=0),
        "tld_length": len(ext.suffix),
    }

def model_vector(raw_url: str) -> list:
    u = normalize_url(raw_url)
    if not u:
        raise ValueError("Invalid URL")
    f = extract_features(u)
    return [f[c] for c in FEATURE_COLUMNS]