from typing import List, Tuple
from app.models import Photo

class QualityScorer:
    """
    Ranks photos in a similarity group by quality.
    Algorithm: highest resolution first, then earliest creation date as tiebreaker.
    Score is deterministic — same inputs always produce same ranking.
    """
    
    @staticmethod
    def calculate_quality_score(photo: Photo) -> float:
        """
        Calculate a deterministic quality score for a photo.
        
        Score components:
        - Primary: megapixels (higher is better)
        - Secondary: creation date (earlier is better, as tiebreaker)
        
        Returns a float score where higher values indicate better quality.
        """
        if not photo.megapixels:
            return 0.0
        
        megapixel_score = photo.megapixels * 1_000_000
        
        date_score = 0.0
        if photo.created_date:
            timestamp = photo.created_date.timestamp()
            date_score = -timestamp / 1_000_000
        
        combined_score = megapixel_score + date_score
        return round(combined_score, 2)
    
    @staticmethod
    def rank_similarity_group(photos: List[Photo]) -> List[Tuple[Photo, float]]:
        """
        Rank photos in a similarity group by quality score.
        
        Args:
            photos: List of Photo objects in the same similarity group
        
        Returns:
            List of (Photo, score) tuples sorted by score descending (best first)
        """
        scored_photos = []
        for photo in photos:
            score = QualityScorer.calculate_quality_score(photo)
            scored_photos.append((photo, score))
        
        scored_photos.sort(key=lambda x: x[1], reverse=True)
        return scored_photos
    
    @staticmethod
    def get_best_photo(photos: List[Photo]) -> Photo:
        """
        Get the best photo from a similarity group.
        """
        if not photos:
            raise ValueError('Cannot rank empty photo list')
        
        ranked = QualityScorer.rank_similarity_group(photos)
        return ranked[0][0]
