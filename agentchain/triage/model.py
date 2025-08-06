def triage_threat(payload: dict):
    threat_type = payload.get("threat_type", "unknown")
    # Simple rule-based severity
    if threat_type in ["ransomware", "critical_vuln"]:
        severity = "critical"
        action = "quarantine"
    elif threat_type in ["malware", "phishing"]:
        severity = "high"
        action = "block"
    elif threat_type in ["suspicious_activity"]:
        severity = "medium"
        action = "alert"
    else:
        severity = "low"
        action = "monitor"
    return {"severity": severity, "action": action, "details": f"Triage for {threat_type}"} 