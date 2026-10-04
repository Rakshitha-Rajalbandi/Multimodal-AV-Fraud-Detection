import re
import ssl
import urllib.parse
import urllib.request
from io import BytesIO

# Configurable List of Target Brands and Official Root Domains
KNOWN_BRANDS = {
    "paypal": ["paypal.com", "paypal.me"],
    "google": ["google.com", "accounts.google.com", "gmail.com", "google.co.uk", "google.ca"],
    "amazon": ["amazon.com", "amazon.co.uk", "amazon.de", "aws.amazon.com"],
    "apple": ["apple.com", "icloud.com"],
    "microsoft": ["microsoft.com", "live.com", "outlook.com", "office.com", "azure.com"],
    "netflix": ["netflix.com"],
    "meta": ["meta.com", "facebook.com", "fb.com", "instagram.com"],
    "chase": ["chase.com"],
    "coinbase": ["coinbase.com"],
    "binance": ["binance.com"],
    "stripe": ["stripe.com"],
    "wellsfargo": ["wellsfargo.com"],
    "bankofamerica": ["bankofamerica.com"],
    "citibank": ["citibank.com"],
    "irs": ["irs.gov"],
    "usps": ["usps.com"],
    "fedex": ["fedex.com"],
    "dhl": ["dhl.com"],
    "linkedin": ["linkedin.com"],
    "twitter": ["twitter.com", "x.com"],
    "ebay": ["ebay.com"],
    "walmart": ["walmart.com"],
    "barclays": ["barclays.com"],
    "hsbc": ["hsbc.com"],
    "steam": ["steampowered.com", "store.steampowered.com"],
    "discord": ["discord.com"],
    "robinhood": ["robinhood.com"],
    "kraken": ["kraken.com"],
    "bybit": ["bybit.com"],
}

# Configurable High-Risk TLD List
HIGH_RISK_TLDS = {
    "xyz", "top", "cheap", "site", "online", "work", "click", "link",
    "club", "run", "tk", "ml", "ga", "cf", "gq", "phishing", "vip",
    "download", "support", "stream", "info", "monster", "kim", "party", "shop"
}

# Leetspeak Substitution Map
LEETSPEAK_MAP = {
    '0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '7': 't', '@': 'a', '$': 's'
}

# Two-part TLD suffixes
TWO_PART_TLDS = {
    "co.uk", "org.uk", "gov.uk", "ac.uk", "co.jp", "com.au", "net.au",
    "org.au", "co.in", "net.in", "gov.in", "org.in", "com.br", "co.nz", "com.tr", "co.za"
}

# Sensitive Authentication Keywords
SENSITIVE_KEYWORDS = [
    "login", "signin", "verify", "verification", "account", "secure",
    "security", "banking", "credential", "password", "wallet", "payment",
    "update", "confirm"
]


def levenshtein_distance(s1, s2):
    """Computes Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def normalize_domain_string(domain_str):
    """Normalizes leetspeak substitutions to reveal underlying brand target."""
    normalized = domain_str.lower()
    for num_char, letter_char in LEETSPEAK_MAP.items():
        normalized = normalized.replace(num_char, letter_char)
    return normalized


def extract_domain_parts(hostname):
    """
    Parses and normalizes hostname into:
    - hostname (full normalized)
    - registered_domain (root domain e.g. google.com or security-update.xyz)
    - root_name (domain name without TLD e.g. google or security-update)
    - tld (top level domain e.g. com or xyz)
    - subdomains (list of subdomain parts e.g. ['accounts'] or ['paypal', 'com', 'login'])
    """
    clean_host = hostname.lower().strip().rstrip(".")
    # Remove port if present
    if ":" in clean_host and not clean_host.startswith("["):
        clean_host = clean_host.split(":")[0]

    # Check IPv4
    if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", clean_host):
        return {
            "hostname": clean_host,
            "registered_domain": clean_host,
            "root_name": clean_host,
            "tld": "",
            "subdomains": [],
            "is_ip": True
        }

    # Check IPv6
    if clean_host.startswith("[") or ":" in clean_host:
        return {
            "hostname": clean_host,
            "registered_domain": clean_host,
            "root_name": clean_host,
            "tld": "",
            "subdomains": [],
            "is_ip": True
        }

    parts = clean_host.split(".")
    if len(parts) == 1:
        return {
            "hostname": clean_host,
            "registered_domain": clean_host,
            "root_name": clean_host,
            "tld": "",
            "subdomains": [],
            "is_ip": False
        }

    # Two-part TLD check (e.g. .co.uk)
    two_part = ".".join(parts[-2:])
    if len(parts) >= 3 and two_part in TWO_PART_TLDS:
        tld = two_part
        root_name = parts[-3]
        registered_domain = f"{root_name}.{tld}"
        subdomains = parts[:-3]
    else:
        tld = parts[-1]
        root_name = parts[-2]
        registered_domain = f"{root_name}.{tld}"
        subdomains = parts[:-2]

    return {
        "hostname": clean_host,
        "registered_domain": registered_domain,
        "root_name": root_name,
        "tld": tld,
        "subdomains": subdomains,
        "is_ip": False
    }


def is_official_domain(parsed_info):
    """
    Checks if registered domain matches any known official brand domain.
    Returns (is_official, brand_name, official_domain_str)
    """
    reg_dom = parsed_info["registered_domain"].lower()
    full_host = parsed_info["hostname"].lower()

    for brand_name, official_domains in KNOWN_BRANDS.items():
        for official in official_domains:
            off_parts = extract_domain_parts(official)
            off_reg = off_parts["registered_domain"].lower()
            if reg_dom == off_reg or full_host == official.lower() or full_host.endswith("." + official.lower()):
                return True, brand_name, official
    return False, None, None


def analyze_brand_impersonation(parsed_info):
    """
    Detects typosquatting, character substitution, and subdomain brand stacking.
    CRITICAL: Only flags when brand appears on an UNAUTHORIZED root domain.
    """
    is_official, brand_name, official_dom = is_official_domain(parsed_info)
    if is_official:
        return {
            "is_impersonating": False,
            "is_official": True,
            "brand": brand_name,
            "official_domain": official_dom,
            "reason": f"Official domain for '{brand_name.capitalize()}'"
        }

    root_name = parsed_info["root_name"]
    normalized_root = normalize_domain_string(root_name)
    subdomains_str = ".".join(parsed_info["subdomains"]).lower()
    normalized_subdomains = normalize_domain_string(subdomains_str)

    # 1. Brand name embedded in root domain (e.g. g00gle-login.xyz or amazon-verify.top)
    for b_name, b_officials in KNOWN_BRANDS.items():
        # Check leetspeak or direct string match in root domain name
        if b_name in normalized_root or b_name in root_name:
            return {
                "is_impersonating": True,
                "is_official": False,
                "brand": b_name,
                "official_domain": b_officials[0],
                "reason": f"Brand Impersonation: Registered domain '{parsed_info['registered_domain']}' imitates '{b_name.capitalize()}'"
            }

        # Check Levenshtein distance on root domain name (e.g. paypa1 vs paypal)
        dist = levenshtein_distance(normalized_root, b_name)
        if 1 <= dist <= 2 and len(root_name) >= 4:
            return {
                "is_impersonating": True,
                "is_official": False,
                "brand": b_name,
                "official_domain": b_officials[0],
                "reason": f"Typosquatting: Domain '{parsed_info['registered_domain']}' is highly similar to '{b_officials[0]}'"
            }

    # 2. Deceptive Subdomain Stacking (e.g. paypal.com.login.security-update.xyz)
    for b_name, b_officials in KNOWN_BRANDS.items():
        if b_name in subdomains_str or b_name in normalized_subdomains:
            return {
                "is_impersonating": True,
                "is_official": False,
                "brand": b_name,
                "official_domain": b_officials[0],
                "reason": f"Deceptive Subdomain Stacking: Brand '{b_name.capitalize()}' hidden in subdomain of unauthorized domain '{parsed_info['registered_domain']}'"
            }

    return {
        "is_impersonating": False,
        "is_official": False,
        "brand": None,
        "official_domain": None,
        "reason": None
    }


def analyze_website_url(url_string):
    """
    Robust Multi-Layered Website Scam & Phishing Detection Engine.
    Combines:
    - URL parsing & root domain normalization
    - Context-aware typosquatting & brand impersonation
    - Registered domain verification
    - High-risk TLD risk weighting
    - HTTPS/SSL protocol checks
    - Raw IP address detection
    - Subdomain depth & hyphen structure inspection
    - Sensitive authentication keyword scanning
    - Safe live HTML inspection (title & password input traps)
    """
    url_input = url_string.strip()
    if not url_input.startswith(("http://", "https://")):
        url_input = "https://" + url_input

    # Parse URL
    try:
        parsed = urllib.parse.urlparse(url_input)
    except Exception as e:
        return {
            "media_type": "Web Domain / URL Link",
            "fraud_risk_score": 0.85,
            "fraud_risk_percent": 85.0,
            "prediction": "FAKE",
            "risk_level": "CRITICAL",
            "confidence": "HIGH",
            "trigger_modality": "MALFORMED URL STRUCTURE",
            "ai_explanation": f"The provided URL could not be parsed safely: {str(e)}",
            "detected_signals": [{"signal": "URL Structure", "status": "FAIL", "icon": "✗", "description": f"Malformed URL: {e}"}]
        }

    scheme = parsed.scheme.lower()
    raw_host = parsed.netloc or parsed.path.split("/")[0]
    path = parsed.path or ""
    query = parsed.query or ""

    parsed_info = extract_domain_parts(raw_host)
    hostname = parsed_info["hostname"]
    reg_domain = parsed_info["registered_domain"]
    tld = parsed_info["tld"]

    signals = []
    risk_points = 0

    # Layer 1: Protocol & HTTPS
    is_https = scheme == "https"
    if is_https:
        signals.append({"signal": "HTTPS Encryption", "status": "PASS", "icon": "✓", "description": "Secure HTTPS connection enforced"})
    else:
        risk_points += 15
        signals.append({"signal": "HTTPS Encryption", "status": "RISK", "icon": "✗", "description": "Unencrypted HTTP scheme used"})

    # Layer 2: IP Address Hostname
    if parsed_info["is_ip"]:
        risk_points += 35
        signals.append({"signal": "Domain Identity", "status": "RISK", "icon": "✗", "description": f"Host uses raw numeric IP address ({hostname})"})
    else:
        signals.append({"signal": "Domain Identity", "status": "PASS", "icon": "✓", "description": f"Valid registered domain name ({reg_domain})"})

    # Layer 3: Brand Verification & Impersonation
    brand_analysis = analyze_brand_impersonation(parsed_info)
    is_official = brand_analysis["is_official"]
    is_impersonating = brand_analysis["is_impersonating"]

    if is_official:
        signals.append({"signal": "Brand Verification", "status": "PASS", "icon": "✓", "description": brand_analysis["reason"]})
    elif is_impersonating:
        risk_points += 45
        signals.append({"signal": "Brand Verification", "status": "RISK", "icon": "✗", "description": brand_analysis["reason"]})
    else:
        signals.append({"signal": "Brand Verification", "status": "PASS", "icon": "✓", "description": "No brand impersonation detected"})

    # HTTPS Signal Correction: PASS only meaningful for verified official domains
    if is_https and not is_official:
        # Overwrite the Layer 1 PASS signal to NEUTRAL for unknown domains
        for sig in signals:
            if sig["signal"] == "HTTPS Encryption" and sig["status"] == "PASS":
                sig["status"] = "NEUTRAL"
                sig["icon"] = "~"
                sig["description"] = "HTTPS present but not a trust indicator for unknown domains"
                break

    # Layer 4: Suspicious TLD
    is_high_risk_tld = tld in HIGH_RISK_TLDS
    if is_high_risk_tld and not is_official:
        risk_points += 25
        signals.append({"signal": "TLD Risk Assessment", "status": "RISK", "icon": "✗", "description": f"High-risk TLD extension (.{tld})"})
    else:
        signals.append({"signal": "TLD Risk Assessment", "status": "PASS", "icon": "✓", "description": f"Standard TLD extension (.{tld if tld else 'N/A'})"})

    # Layer 5: Subdomain & Domain Structure
    sub_count = len(parsed_info["subdomains"])
    hyphen_count = hostname.count("-")
    
    if sub_count > 2 and not is_official:
        risk_points += 15
        signals.append({"signal": "Subdomain Structure", "status": "RISK", "icon": "✗", "description": f"Excessive subdomain depth ({sub_count} levels)"})
    else:
        signals.append({"signal": "Subdomain Structure", "status": "PASS", "icon": "✓", "description": f"Normal subdomain structure ({sub_count} levels)"})

    if hyphen_count >= 2 and not is_official:
        risk_points += 15
        signals.append({"signal": "Hostname Hyphens", "status": "RISK", "icon": "✗", "description": f"Multiple hyphens in domain ({hyphen_count} hyphens)"})

    # Layer 6: Sensitive Authentication Keywords
    combined_path_query = (hostname + "/" + path + "?" + query).lower()
    found_keywords = [kw for kw in SENSITIVE_KEYWORDS if kw in combined_path_query]

    if found_keywords and not is_official:
        risk_points += 20
        signals.append({"signal": "Path Keyword Inspection", "status": "RISK", "icon": "✗", "description": f"Sensitive login keywords in URL: {', '.join(found_keywords)}"})
    elif found_keywords and is_official:
        signals.append({"signal": "Path Keyword Inspection", "status": "PASS", "icon": "✓", "description": f"Authentication path on official domain ({', '.join(found_keywords)})"})

    # Layer 7: Live HTML Inspection (Safe Fetch)
    html_title = None
    has_password_trap = False
    cloned_title_detected = False

    try:
        req = urllib.request.Request(
            url_input,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(req, timeout=4, context=ctx) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            if "html" in content_type:
                html_bytes = response.read(100000)  # Read first 100KB safely
                html_text = html_bytes.decode("utf-8", errors="ignore")

                title_match = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
                if title_match:
                    html_title = title_match.group(1).strip()

                if re.search(r'<input[^>]+type=["\']password["\']', html_text, re.IGNORECASE):
                    has_password_trap = True

                if html_title and not is_official:
                    for b_name, b_officials in KNOWN_BRANDS.items():
                        if b_name in html_title.lower():
                            cloned_title_detected = True
                            risk_points += 40
                            signals.append({
                                "signal": "HTML Title Inspection",
                                "status": "RISK",
                                "icon": "✗",
                                "description": f"Duplicate Website Clone: Title claims '{b_name.capitalize()}' on unauthorized domain '{reg_domain}'"
                            })
                            break
    except Exception:
        # Unreachable site — add a modest risk penalty (phishing sites are often unreliable/ephemeral)
        risk_points += 10

    if has_password_trap and not is_official:
        risk_points += 30
        signals.append({
            "signal": "Credential Input Protection",
            "status": "RISK",
            "icon": "✗",
            "description": "Credential Trap: Password input form hosted on unverified domain"
        })

    # Calibrated Risk Scoring (0–100)
    # If official brand domain, enforce low risk score cap (< 30)
    if is_official:
        final_score = min(15, risk_points)
    else:
        final_score = min(98, max(5, risk_points))

    fraud_risk_score = round(final_score / 100.0, 4)
    fraud_risk_percent = round(float(final_score), 2)

    # Classification Threshold Calibration
    # 0 - 19 -> GENUINE / LOW RISK (REAL)
    # 20 - 44 -> SUSPICIOUS / REVIEW (SUSPICIOUS)
    # 45 - 100 -> HIGH RISK / LIKELY SCAM (FAKE)
    if final_score >= 45:
        prediction = "FAKE"
        classification_label = "LIKELY SCAM"
        risk_level = "CRITICAL"
    elif final_score >= 20:
        prediction = "SUSPICIOUS"
        classification_label = "SUSPICIOUS"
        risk_level = "MEDIUM"
    else:
        prediction = "REAL"
        classification_label = "GENUINE"
        risk_level = "LOW"

    # Natural Language Explainable AI Rationale
    risk_signals_list = [s["description"] for s in signals if s["status"] == "RISK"]
    if classification_label == "LIKELY SCAM":
        ai_explanation = (
            f"High Risk ({fraud_risk_percent}/100) because the URL exhibits malicious scam indicators: "
            f"{'; '.join(risk_signals_list[:3])}."
        )
    elif classification_label == "SUSPICIOUS":
        ai_explanation = (
            f"Suspicious ({fraud_risk_percent}/100) due to potential risk signals: "
            f"{'; '.join(risk_signals_list[:2])}. Exercise caution before entering credentials."
        )
    else:
        ai_explanation = (
            f"Genuine website ({fraud_risk_percent}/100 Risk). "
            f"The URL domain ({reg_domain}) exhibits legitimate security compliance and brand integrity."
        )

    return {
        "media_type": "Web Domain / URL Link",
        "fraud_risk_score": fraud_risk_score,
        "fraud_risk_percent": fraud_risk_percent,
        "prediction": prediction,
        "classification": classification_label,
        "risk_level": risk_level,
        "confidence": "HIGH" if abs(final_score - 45) > 15 else "MEDIUM",
        "trigger_modality": f"URL FORENSICS ({classification_label})",
        "ai_explanation": ai_explanation,
        "detected_signals": signals,
        "domain_analysis": {
            "hostname": hostname,
            "registered_domain": reg_domain,
            "scheme": scheme,
            "is_https": is_https,
            "is_ip_address": parsed_info["is_ip"],
            "impersonated_brand": brand_analysis["brand"].capitalize() if brand_analysis["brand"] else None,
            "official_domain": brand_analysis["official_domain"],
            "html_title": html_title,
            "phishing_indicators": risk_signals_list
        },
        "video": {"prediction": prediction, "fake_probability": fraud_risk_score, "real_probability": round(1.0 - fraud_risk_score, 4), "visual_score_percent": fraud_risk_percent, "frames_evaluated": 0, "heatmaps": []},
        "audio": {"prediction": "N/A", "fake_probability": 0.0, "real_probability": 1.0, "audio_score_percent": 0.0},
        "sync": {"sync_mismatch_score": 0.0},
        "fusion": {"fraud_risk_percent": fraud_risk_percent, "prediction": prediction},
        "heatmaps": [],
        "timeline_risk": [
            {"timestamp_sec": 0.0, "timestamp_str": "Domain Check", "risk_percent": fraud_risk_percent},
            {"timestamp_sec": 0.5, "timestamp_str": "SSL Protocol", "risk_percent": 10.0 if is_https else 75.0},
            {"timestamp_sec": 1.0, "timestamp_str": "Brand Verification", "risk_percent": 90.0 if is_impersonating else 10.0},
            {"timestamp_sec": 1.5, "timestamp_str": "HTML Inspection", "risk_percent": fraud_risk_percent}
        ],
        "suspicious_timestamps": [{
            "timestamp_str": "URL Forensic",
            "risk_percent": fraud_risk_percent,
            "description": desc
        } for desc in risk_signals_list]
    }
