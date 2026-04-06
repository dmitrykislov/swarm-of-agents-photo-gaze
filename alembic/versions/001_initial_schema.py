"""Initial schema creation with photos, embeddings, processing_state, and user_preferences tables.

Revision ID: 001
Revises:
Create Date: 2026-04-04

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial schema with all required tables and indexes."""
    # Create user_preferences table first (no foreign key dependencies)
    op.create_table(
        "user_preferences",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("username", sa.String(length=100), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("preferred_embedding_model", sa.String(length=100), nullable=False, server_default="clip-vit-base-patch32"),
        sa.Column("enable_auto_processing", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
        sa.UniqueConstraint("email"),
    )
    op.create_index("idx_user_preferences_username", "user_preferences", ["username"])
    op.create_index("idx_user_preferences_email", "user_preferences", ["email"])

    # Create photos table
    op.create_table(
        "photos",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("file_path", sa.String(length=512), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("megapixels", sa.Float(), nullable=True),
        sa.Column("created_date", sa.DateTime(), nullable=True),
        sa.Column("modified_date", sa.DateTime(), nullable=False),
        sa.Column("camera_info", sa.String(length=255), nullable=True),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("similarity_group_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("file_path"),
    )
    op.create_index("idx_photos_filename", "photos", ["filename"])
    op.create_index("idx_photos_similarity_group_id", "photos", ["similarity_group_id"])
    op.create_index("idx_photos_quality_score", "photos", ["quality_score"])

    # Create embeddings table
    op.create_table(
        "embeddings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("photo_id", sa.Integer(), nullable=False),
        sa.Column("embedding_model", sa.String(length=100), nullable=False),
        sa.Column("vector_dimension", sa.Integer(), nullable=False),
        sa.Column("qdrant_point_id", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["photo_id"], ["photos.id"], ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("qdrant_point_id"),
    )
    op.create_index("idx_embeddings_photo_id", "embeddings", ["photo_id"])
    op.create_index("idx_embeddings_model", "embeddings", ["embedding_model"])
    op.create_index("idx_embeddings_qdrant_point_id", "embeddings", ["qdrant_point_id"])

    # Create processing_state table
    op.create_table(
        "processing_state",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("photo_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("extraction_status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("embedding_status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["photo_id"], ["photos.id"], ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("photo_id"),
    )
    op.create_index("idx_processing_state_photo_id", "processing_state", ["photo_id"])
    op.create_index("idx_processing_state_status", "processing_state", ["status"])
    op.create_index("idx_processing_state_updated_at", "processing_state", ["updated_at"])


def downgrade() -> None:
    """Drop all tables and indexes created in upgrade."""
    op.drop_index("idx_processing_state_updated_at", table_name="processing_state")
    op.drop_index("idx_processing_state_status", table_name="processing_state")
    op.drop_index("idx_processing_state_photo_id", table_name="processing_state")
    op.drop_table("processing_state")

    op.drop_index("idx_embeddings_qdrant_point_id", table_name="embeddings")
    op.drop_index("idx_embeddings_model", table_name="embeddings")
    op.drop_index("idx_embeddings_photo_id", table_name="embeddings")
    op.drop_table("embeddings")

    op.drop_index("idx_photos_quality_score", table_name="photos")
    op.drop_index("idx_photos_similarity_group_id", table_name="photos")
    op.drop_index("idx_photos_filename", table_name="photos")
    op.drop_table("photos")

    op.drop_index("idx_user_preferences_email", table_name="user_preferences")
    op.drop_index("idx_user_preferences_username", table_name="user_preferences")
    op.drop_table("user_preferences")

