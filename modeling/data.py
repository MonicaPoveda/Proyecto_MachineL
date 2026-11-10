"""Dataset construction for the phishing classifier.

Preferred sources (offline, already present in the repository):
  * malicious   -> static/datasets/indicators_of_compromise_ioc_20261106.csv
  * legitimate  -> static/datasets/top-1m.csv

If those files are missing (fresh clone, no internet), a synthetic fallback
generator produces a well-formed mock dataset so the pipeline remains testable.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from modeling.features import normalize_url

IOC_URL_TYPES = {"url", "fqdn", "domain", "ip address", "ipv4"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_malicious_urls(path: Path, limit: int | None = None) -> list[str]:
    data = pd.read_csv(path, usecols=["value", "type"])
    data["type"] = data["type"].astype(str).str.lower()
    mask = data["type"].isin(IOC_URL_TYPES) & data["value"].notna()
    values = data.loc[mask, "value"].astype(str).str.strip()
    values = values[values != ""]
    if limit:
        values = values.head(limit)

    urls: list[str] = []
    for value in values:
        normalized = normalize_url(value)
        if normalized:
            urls.append(normalized)
    return urls


def load_legitimate_urls(path: Path, limit: int | None = None, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    data = pd.read_csv(path, header=None, usecols=[1], names=["domain"], nrows=limit, dtype=str)
    domains = data["domain"].astype(str).str.strip().str.lower()
    domains = domains[domains.str.match(r"^[a-z0-9.\-]+$")]
    if limit:
        domains = domains.head(limit)

    urls: list[str] = []
    for domain in domains:
        scheme = rng.choice(_LEGIT_SCHEMES)
        host = f"www.{domain}" if rng.random() < 0.4 else domain
        path = rng.choice(_LEGIT_PATHS)
        query = rng.choice(_LEGIT_QUERIES)
        normalized = normalize_url(f"{scheme}://{host}{path}{query}")
        if normalized:
            urls.append(normalized)
    return urls


_LEGIT_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "wikipedia.org", "amazon.com",
    "twitter.com", "instagram.com", "linkedin.com", "reddit.com", "netflix.com",
    "github.com", "microsoft.com", "apple.com", "cloudflare.com", "mozilla.org",
    "stackoverflow.com", "bbc.com", "cnn.com", "nytimes.com", "ebay.com",
    "paypal.com", "spotify.com", "dropbox.com", "adobe.com", "oracle.com",
    "salesforce.com", "ibm.com", "intel.com", "nvidia.com", "zoom.us",
]
_LEGIT_PATHS = [
    "", "/search", "/watch", "/account", "/settings", "/help", "/about",
    "/products", "/login", "/signin", "/docs", "/blog", "/news", "/support",
    "/downloads", "/api/v1", "/store", "/pricing", "/careers", "/contact",
]
_LEGIT_QUERIES = ["", "", "", "?q=example", "?page=1", "?id=42", "?lang=en", "?ref=home"]
_LEGIT_SCHEMES = ["https", "https", "https", "http"]

_BRANDS = ["paypal", "google", "microsoft", "apple", "amazon", "netflix",
           "facebook", "bank", "secure", "login"]
_RISKY_TLDS = ["tk", "ml", "ga", "cf", "top", "xyz", "icu", "click", "work", "buzz"]


def _random_ip(rng: random.Random) -> str:
    return f"{rng.randint(1, 254)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def _random_token(rng: random.Random, length: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(rng.choice(alphabet) for _ in range(length))


def _synthetic_legit(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    urls: set[str] = set()
    while len(urls) < count:
        domain = rng.choice(_LEGIT_DOMAINS)
        scheme = rng.choice(_LEGIT_SCHEMES)
        host = f"www.{domain}" if rng.random() < 0.4 else domain
        path = rng.choice(_LEGIT_PATHS)
        query = rng.choice(_LEGIT_QUERIES)
        urls.add(f"{scheme}://{host}{path}{query}")
    return list(urls)


def _synthetic_malicious(count: int, seed: int) -> list[str]:
    rng = random.Random(seed + 1)
    urls: set[str] = set()
    while len(urls) < count:
        kind = rng.randint(0, 4)
        if kind == 0:  # brand impersonation on a risky TLD
            brand = rng.choice(_BRANDS)
            verb = rng.choice(["verify", "login", "secure", "account", "update"])
            tld = rng.choice(_RISKY_TLDS)
            urls.add(f"http://{brand}-{verb}-{_random_token(rng, 7)}.{tld}/login")
        elif kind == 1:  # raw IP host with a non-standard port
            port = rng.choice(["8080", "3128", "8000", "8081"])
            urls.add(f"http://{_random_ip(rng)}:{port}/{rng.choice(['login', 'update', 'pay', 'verify'])}")
        elif kind == 2:  # userinfo / at-sign abuse
            urls.add(f"http://{rng.choice(_LEGIT_DOMAINS)}@{rng.choice(_BRANDS)}-{rng.randint(1000, 9999)}.{rng.choice(_RISKY_TLDS)}/login")
        elif kind == 3:  # long random host + executable payload
            urls.add(f"http://{_random_token(rng, rng.randint(20, 40))}.{rng.choice(_RISKY_TLDS)}/{rng.randint(1000, 9999)}.exe")
        else:  # punycode host
            urls.add(f"http://xn--{_random_token(rng, 6)}.{rng.choice(_RISKY_TLDS)}/account")
    return list(urls)


def build_dataset(
    malicious_limit: int = 5000,
    legit_limit: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a balanced, shuffled DataFrame with columns ``url`` and ``label``."""
    root = _project_root()
    ioc_path = root / "static" / "datasets" / "indicators_of_compromise_ioc_20261106.csv"
    top_path = root / "static" / "datasets" / "top-1m.csv"

    if ioc_path.is_file() and top_path.is_file():
        print(f"Using local datasets:\n  - {ioc_path.name}\n  - {top_path.name}")
        malicious = load_malicious_urls(ioc_path, malicious_limit)
        legitimate = load_legitimate_urls(top_path, legit_limit, seed=random_state)
    else:
        print("Local datasets not found; generating the synthetic fallback dataset.")
        malicious = _synthetic_malicious(malicious_limit, random_state)
        legitimate = _synthetic_legit(legit_limit, random_state)

    if not malicious or not legitimate:
        raise RuntimeError("Could not build a non-empty dataset from either source.")

    frame = pd.DataFrame(
        {"url": malicious + legitimate, "label": [1] * len(malicious) + [0] * len(legitimate)}
    ).drop_duplicates(subset="url")

    min_count = min(int((frame["label"] == 1).sum()), int((frame["label"] == 0).sum()))
    frame = pd.concat(
        [
            frame[frame["label"] == 1].sample(min_count, random_state=random_state),
            frame[frame["label"] == 0].sample(min_count, random_state=random_state),
        ]
    ).sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(
        f"Balanced dataset: {len(frame)} URLs "
        f"({int((frame['label'] == 1).sum())} phishing, "
        f"{int((frame['label'] == 0).sum())} legitimate)."
    )
    return frame
