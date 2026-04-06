from app import db
from datetime import datetime

class Photo(db.Model):
    __tablename__ = 'photos'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(512), nullable=False, unique=True)
    file_size_bytes = db.Column(db.Integer, nullable=False)
    width = db.Column(db.Integer, nullable=True)
    height = db.Column(db.Integer, nullable=True)
    megapixels = db.Column(db.Float, nullable=True)
    created_date = db.Column(db.DateTime, nullable=True)
    modified_date = db.Column(db.DateTime, nullable=False)
    camera_info = db.Column(db.String(255), nullable=True)
    quality_score = db.Column(db.Float, nullable=True)
    similarity_group_id = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Photo {self.id}: {self.filename}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'file_path': self.file_path,
            'file_size_bytes': self.file_size_bytes,
            'width': self.width,
            'height': self.height,
            'megapixels': self.megapixels,
            'created_date': self.created_date.isoformat() if self.created_date else None,
            'modified_date': self.modified_date.isoformat() if self.modified_date else None,
            'camera_info': self.camera_info,
            'quality_score': self.quality_score,
            'similarity_group_id': self.similarity_group_id
        }
