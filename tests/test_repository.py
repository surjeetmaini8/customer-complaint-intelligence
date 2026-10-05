import datetime as dt

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base
from src.database import repository as repo

def test_count_uses_same_filters_as_list():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as db:
        repo.upsert_complaint(db, {
            "complaint_id": "CMP-1", "source": "email", "original_text": "refund is late",
            "processed_text": "refund is late", "category": "Refund", "severity": "HIGH",
            "sentiment": "Negative", "region": "North India", "created_at": dt.datetime(2024,1,1),
        })
        repo.upsert_complaint(db, {
            "complaint_id": "CMP-2", "source": "chat", "original_text": "refund question",
            "processed_text": "refund question", "category": "Refund", "severity": "LOW",
            "sentiment": "Neutral", "region": "South India", "created_at": dt.datetime(2024,1,2),
        })
        rows = repo.list_complaints(db, category="Refund", region="North India", search="late")
        total = repo.count_complaints(db, category="Refund", region="North India", search="late")
        assert len(rows) == 1
        assert total == 1
