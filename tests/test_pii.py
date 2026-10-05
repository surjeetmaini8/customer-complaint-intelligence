from src.preprocessing.pii import detect_pii, mask_pii

def test_pii_masking_removes_email_and_phone():
    text = "Email me at user@example.com or call +91 9876543210"
    masked = mask_pii(text)
    assert "user@example.com" not in masked
    assert "9876543210" not in masked
    labels = {x["label"] for x in detect_pii(text)}
    assert {"EMAIL", "PHONE"}.issubset(labels)
