"""Tests for SQLite support in make_engine (the native app's database)."""
import threading

from sqlalchemy.orm import sessionmaker

from app.database import make_engine
from app.models import Base, Photo


def test_sqlite_sets_pragmas(tmp_path):
    """WAL + busy timeout + foreign keys are applied to every connection."""
    eng = make_engine(f"sqlite:///{tmp_path / 'pragmas.db'}")
    with eng.connect() as c:
        assert c.exec_driver_sql("PRAGMA busy_timeout").scalar() == 5000
        assert c.exec_driver_sql("PRAGMA journal_mode").scalar().lower() == "wal"
        assert c.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1


def test_sqlite_usable_across_threads(tmp_path):
    """The job queue runs DB work in threads; check_same_thread=False must
    allow a session created on another thread to use the engine."""
    eng = make_engine(f"sqlite:///{tmp_path / 'threads.db'}")
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng)

    s = Session()
    s.add(Photo(filename="a.jpg", file_path="/a.jpg", file_size=1, mime_type="image/jpeg"))
    s.commit()
    s.close()

    result = {}
    def worker():
        s2 = Session()
        try:
            result["count"] = s2.query(Photo).count()
        finally:
            s2.close()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert result["count"] == 1


def test_sqlite_schema_round_trips(tmp_path):
    """All models create on SQLite (no Postgres-only types) and a Photo +
    cascade rows persist — confirms the native app's schema works."""
    from app.models import Embedding, ProcessingState
    eng = make_engine(f"sqlite:///{tmp_path / 'schema.db'}")
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng)
    s = Session()
    p = Photo(filename="x.jpg", file_path="/x.jpg", file_size=10, mime_type="image/jpeg")
    s.add(p)
    s.commit()
    s.add_all([
        Embedding(photo_id=p.id, embedding_model="dinov2_vits14", vector_dimension=384),
        ProcessingState(photo_id=p.id, status="completed"),
    ])
    s.commit()
    assert s.query(Photo).count() == 1
    assert s.query(Embedding).count() == 1
    s.close()
