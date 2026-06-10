"""Database initialization and Qdrant collection setup.

Supports Postgres (Docker) and SQLite (native, single-user desktop app) from
the same DATABASE_URL — make_engine applies the right options for each.
"""
import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.models import Base


def make_engine(url: str, **kwargs):
    """Create an SQLAlchemy engine with the right options per backend.

    SQLite (native app): allow cross-thread use (the job queue runs work in
    threads) and set pragmas — WAL for better read/write concurrency and a
    busy timeout so brief lock contention waits instead of raising
    "database is locked". Postgres is left exactly as before.
    """
    is_sqlite = url.startswith("sqlite")
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    eng = create_engine(url, connect_args=connect_args, **kwargs)
    if is_sqlite:
        @event.listens_for(eng, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=5000")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
    return eng


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/app_db"
)
engine = make_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables from SQLAlchemy models."""
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized")


def init_qdrant_collection():
    """Initialize the Qdrant collection if it doesn't exist.

    Vectors are 384-dimensional to match DINOv2 ViT-S/14 (see
    EmbeddingGenerator.embedding_dim). Do NOT change this to 1024 without
    also swapping the model — a mismatch makes every upsert fail."""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = "embeddings"
    vector_size = 384
    
    try:
        client = QdrantClient(url=qdrant_url)
        
        # Check if collection exists
        try:
            client.get_collection(collection_name)
        except Exception:
            # Collection doesn't exist, create it
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"Created Qdrant collection '{collection_name}' with {vector_size}-dim vectors")
    except Exception as e:
        print(f"Warning: Failed to initialize Qdrant collection: {e}")

