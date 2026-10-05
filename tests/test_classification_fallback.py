from src.classification.inference import ClassificationService

def test_missing_subcategory_model_requires_human_review(monkeypatch):
    service = ClassificationService.__new__(ClassificationService)
    service._subcategory_models = {}
    result = service.predict_subcategory("some complaint", "Unknown Category")
    assert result["confidence"] == 0.0
    assert result["needs_human_review"] is True
    assert result["error"] == "no_subcategory_model"
