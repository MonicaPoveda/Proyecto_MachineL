"""Lexical and syntactic URL feature extraction (fresh implementation).

All features are computed statically from the URL string (no network access,
no page execution), matching the project's CRISP-DM business constraints.
The extractor is deterministic and self-contained.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlsplit

import tldextract

# Offline extractor: uses the public-suffix snapshot bundled with tldextract,
# so nothing is downloaded at import time.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "s.id",
}

# Matched against whole hostname tokens (split on non-alphanumeric chars),
# NOT as raw substrings, to avoid flagging legitimate URLs such as
# "accounts.google.com" or "support.microsoft.com".
SUSPICIOUS_TOKENS = {
    "login", "signin", "verify", "verification", "secure", "account",
    "update", "confirm", "password", "wallet", "banking", "billing",
    "invoice", "suspend", "suspended", "unlock", "token", "auth",
    "bonus", "prize", "gift", "support",
}

EXECUTABLE_EXT = (
    ".exe", ".dll", ".sh", ".bat", ".ps1", ".vbs", ".hta", ".apk",
    ".msi", ".jar", ".scr", ".zip", ".rar", ".bin",
)

RISKY_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "top", "xyz", "icu", "cyou", "click",
    "work", "zip", "mov", "buzz", "rest", "monster", "stream", "quest",
}

# Well-known brands used only for an impersonation heuristic (a brand label
# appearing under someone else's registered domain).
BRAND_TOKENS = {
    "google", "youtube", "facebook", "instagram", "whatsapp", "amazon",
    "apple", "microsoft", "outlook", "office", "netflix", "paypal",
    "ebay", "linkedin", "dropbox", "spotify", "gmail", "yahoo",
    "santander", "bancolombia", "bbva", "davivienda", "wells", "chase",
    "citi", "bank",
}

_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")

FEATURE_COLUMNS = [
    "url_length", "host_length", "path_length", "query_length",
    "n_dots", "n_hyphens", "n_underscores", "n_slashes", "n_digits",
    "digit_ratio", "letter_ratio", "n_special_chars",
    "n_subdomains", "path_depth", "n_query_params",
    "has_https", "has_ip_host", "has_port", "port_is_nonstandard",
    "has_at_symbol", "has_double_slash_in_path", "has_punycode",
    "is_shortener", "has_executable_ext", "risky_tld", "tld_length",
    "host_entropy", "url_entropy", "longest_host_token",
    "n_suspicious_tokens", "brand_in_subdomain",
]


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def normalize_url(raw: str) -> str:
    """Return a canonical URL with a scheme, or an empty string when invalid.

    Unlike the previous implementation, this one preserves the scheme so the
    ``has_https`` feature carries real signal.
    """
    if raw is None:
        return ""
    value = str(raw).strip().strip('"').strip("'")
    value = value.replace("hxxp://", "http://").replace("hxxps://", "https://")
    value = value.replace("[.]", ".")
    if not value:
        return ""
    if not _SCHEME.match(value):
        value = "http://" + value
    try:
        parsed = urlsplit(value)
    except ValueError:
        return ""
    host = (parsed.hostname or "").rstrip(".").lower()
    if not host:
        return ""
    scheme = "https" if parsed.scheme == "https" else "http"
    netloc = host
    if "@" in (parsed.netloc or ""):
        netloc = "u@" + netloc
    if parsed.port:
        netloc += f":{parsed.port}"
    return f"{scheme}://{netloc}{parsed.path or ''}{'?' + parsed.query if parsed.query else ''}"


def extract_features(url: str) -> dict[str, float]:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    port = parsed.port

    body = url.split("://", 1)[1] if "://" in url else url
    body_len = max(len(body), 1)

    is_ip = bool(_IPV4.match(host))
    if is_ip:
        domain, suffix, subdomain = "", "", ""
    else:
        ext = _EXTRACT(host)
        domain = ext.domain or ""
        suffix = ext.suffix or ""
        subdomain = ext.subdomain or ""
    registered = ".".join(part for part in (domain, suffix) if part) or host

    n_subdomains = 0 if is_ip else len([s for s in subdomain.split(".") if s])

    n_digits = sum(ch.isdigit() for ch in body)
    n_letters = sum(ch.isalpha() for ch in body)

    host_tokens = _tokens(host)
    n_suspicious = sum(1 for token in host_tokens if token in SUSPICIOUS_TOKENS)

    brand_in_subdomain = 0
    if not is_ip and subdomain:
        sub_tokens = _tokens(subdomain)
        if any(brand in sub_tokens and domain != brand for brand in BRAND_TOKENS):
            brand_in_subdomain = 1

    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "n_dots": body.count("."),
        "n_hyphens": body.count("-"),
        "n_underscores": body.count("_"),
        "n_slashes": body.count("/"),
        "n_digits": n_digits,
        "digit_ratio": n_digits / body_len,
        "letter_ratio": n_letters / body_len,
        "n_special_chars": len(re.findall(r"[^A-Za-z0-9./:\-_]", body)),
        "n_subdomains": n_subdomains,
        "path_depth": len([s for s in path.split("/") if s]),
        "n_query_params": len([q for q in query.split("&") if q]),
        "has_https": int(url.startswith("https://")),
        "has_ip_host": int(is_ip),
        "has_port": int(port is not None),
        "port_is_nonstandard": int(port is not None and port not in (80, 443)),
        "has_at_symbol": int("@" in (parsed.netloc or "")),
        "has_double_slash_in_path": int("//" in path),
        "has_punycode": int("xn--" in host),
        "is_shortener": int(registered in SHORTENERS),
        "has_executable_ext": int(path.lower().endswith(EXECUTABLE_EXT)),
        "risky_tld": int(suffix.split(".")[-1] in RISKY_TLDS if suffix else 0),
        "tld_length": len(suffix),
        "host_entropy": _entropy(host),
        "url_entropy": _entropy(body),
        "longest_host_token": max((len(t) for t in re.split(r"[.\-]", host)), default=0),
        "n_suspicious_tokens": n_suspicious,
        "brand_in_subdomain": brand_in_subdomain,
    }


def feature_vector(raw_url: str) -> list[float]:
    """Return the ordered numeric feature vector for a raw URL."""
    normalized = normalize_url(raw_url)
    if not normalized:
        raise ValueError(f"Invalid URL (no valid host): {raw_url!r}")
    features = extract_features(normalized)
    return [features[column] for column in FEATURE_COLUMNS]
