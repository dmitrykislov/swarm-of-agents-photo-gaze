"""Database initialization and Qdrant collection setup."""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.models import Base


# PostgreSQL database setup
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/app_db"
)
engine = create_engine(DATABASE_URL)
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

