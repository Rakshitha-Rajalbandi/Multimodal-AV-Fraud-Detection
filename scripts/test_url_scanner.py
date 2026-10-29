import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.url_scanner import analyze_website_url

GENUINE_TEST_CASES = [
    "https://www.google.com",
    "https://accounts.google.com",
    "https://www.paypal.com",
    "https://www.amazon.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://github.com",
    "https://www.wikipedia.org"
]

SUSPICIOUS_TEST_CASES = [
    "https://paypa1.com",
    "https://g00gle-login.xyz",
    "https://amazon-verify.top/login",
    "https://paypal.com.login.security-update.xyz",
    "https://google.com.verify-account.top",
    "http://192.168.1.100/login",
    "https://secure-login-verification.xyz/account",
    "https://paypal-login-verification.top"
]

def run_tests():
    print("=" * 85)
    print("URL FORENSIC SCANNER - SYNTHETIC BENCHMARK TEST SUITE")
    print("=" * 85)

    genuine_passed = 0
    genuine_failed = 0

    print("\n--- 1. BENCHMARKING GENUINE WEBSITES (MUST BE < 30 RISK / GENUINE) ---")
    for url in GENUINE_TEST_CASES:
        res = analyze_website_url(url)
        score = res["fraud_risk_percent"]
        pred = res["prediction"]
        label = res["classification"]
        explanation = res["ai_explanation"]
        
        is_pass = pred == "REAL" and score < 30.0
        if is_pass:
            genuine_passed += 1
            status_str = "[PASS] (GENUINE)"
        else:
            genuine_failed += 1
            status_str = "[FAIL] (FALSE POSITIVE)"

        print(f"\nURL: {url}")
        print(f"  Result: {status_str} | Risk Score: {score:.1f}/100 | Classification: {label}")
        print(f"  Explanation: {explanation}")

    print("\n" + "=" * 85)
    scam_passed = 0
    scam_failed = 0

    print("--- 2. BENCHMARKING SUSPICIOUS / SCAM WEBSITES (MUST BE >= 30 RISK) ---")
    for url in SUSPICIOUS_TEST_CASES:
        res = analyze_website_url(url)
        score = res["fraud_risk_percent"]
        pred = res["prediction"]
        label = res["classification"]
        explanation = res["ai_explanation"]

        is_pass = pred in ["FAKE", "SUSPICIOUS"] and score >= 30.0
        if is_pass:
            scam_passed += 1
            status_str = "[PASS] (DETECTED)"
        else:
            scam_failed += 1
            status_str = "[FAIL] (MISSED SCAM)"

        print(f"\nURL: {url}")
        print(f"  Result: {status_str} | Risk Score: {score:.1f}/100 | Classification: {label}")
        print(f"  Explanation: {explanation}")
        risk_signals = [s['description'] for s in res['detected_signals'] if s['status'] == 'RISK']
        if risk_signals:
            print(f"  Risk Signals: {', '.join(risk_signals)}")

    print("\n" + "=" * 85)
    print("TEST SUITE SUMMARY")
    print("=" * 85)
    print(f"Genuine URLs Correctly Classified:   {genuine_passed}/{len(GENUINE_TEST_CASES)} ({genuine_passed/len(GENUINE_TEST_CASES)*100:.1f}%)")
    print(f"Suspicious URLs Correctly Detected:  {scam_passed}/{len(SUSPICIOUS_TEST_CASES)} ({scam_passed/len(SUSPICIOUS_TEST_CASES)*100:.1f}%)")
    print("=" * 85)

    if genuine_failed == 0 and scam_failed == 0:
        print("\nSUCCESS: ALL TEST CASES PASSED WITH 100% ACCURACY AND 0 FALSE POSITIVES!")
    else:
        print(f"\nNOTICE: {genuine_failed} false positives and {scam_failed} missed scams.")

if __name__ == "__main__":
    run_tests()
