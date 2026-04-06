import React, { useState } from 'react';
import axios from 'axios';

/**
 * SimilarityGroupCard displays a similarity group with checkboxes for each image.
 * The best image (highest quality_score) is pre-selected by default.
 * Users can toggle checkboxes and click 'Deduplicate' to move unchecked photos to trash.
 */
export function SimilarityGroupCard({ group, onDeduplicateSuccess }) {
  const [selectedPhotoIds, setSelectedPhotoIds] = useState(() => {
    // Pre-select all photos except the best one (highest quality_score)
    if (!group.members || group.members.length === 0) return new Set();
    
    const bestPhoto = group.members.reduce((best, current) => {
      const bestScore = best.quality_score || 0;
      const currentScore = current.quality_score || 0;
      return currentScore > bestScore ? current : best;
    });
    
    const selected = new Set();
    group.members.forEach(photo => {
      if (photo.photo_id !== bestPhoto.photo_id) {
        selected.add(photo.photo_id);
      }
    });
    return selected;
  });
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const togglePhotoSelection = (photoId) => {
    const newSelected = new Set(selectedPhotoIds);
    if (newSelected.has(photoId)) {
      newSelected.delete(photoId);
    } else {
      newSelected.add(photoId);
    }
    setSelectedPhotoIds(newSelected);
  };
  
  const handleDeduplicate = async () => {
    if (selectedPhotoIds.size === 0) {
      setError('Please select at least one photo to deduplicate');
      return;
    }
    
    const confirmMessage = `Move ${selectedPhotoIds.size} photo(s) to trash?`;
    if (!window.confirm(confirmMessage)) {
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('/deduplicate', {
        photo_ids: Array.from(selectedPhotoIds)
      });
      
      if (response.status === 200) {
        // Refresh the group after successful deduplication
        if (onDeduplicateSuccess) {
          onDeduplicateSuccess(group.group_id);
        }
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Deduplication failed');
      console.error('Deduplication error:', err);
    } finally {
      setIsLoading(false);
    }
  };
  
  if (!group.members || group.members.length === 0) {
    return <div className="group-card">No photos in group</div>;
  }
  
  const bestPhoto = group.members.reduce((best, current) => {
    const bestScore = best.quality_score || 0;
    const currentScore = current.quality_score || 0;
    return currentScore > bestScore ? current : best;
  });
  
  return (
    <div className="similarity-group-card">
      <div className="group-header">
        <h3>Group {group.group_id}</h3>
        <p>Similarity: {(group.similarity_score * 100).toFixed(1)}%</p>
      </div>
      
      <div className="photos-grid">
        {group.members.map(photo => (
          <div key={photo.photo_id} className="photo-item">
            <div className="photo-checkbox-wrapper">
              <input
                type="checkbox"
                id={`photo-${photo.photo_id}`}
                checked={selectedPhotoIds.has(photo.photo_id)}
                onChange={() => togglePhotoSelection(photo.photo_id)}
                disabled={photo.photo_id === bestPhoto.photo_id}
              />
              <label htmlFor={`photo-${photo.photo_id}`}>
                {photo.photo_id === bestPhoto.photo_id ? '(Best)' : 'Remove'}
              </label>
            </div>
            {photo.thumbnail && (
              <img src={photo.thumbnail} alt={photo.filename} className="photo-thumbnail" />
            )}
            <p className="photo-filename">{photo.filename}</p>
            {photo.quality_score !== undefined && (
              <p className="photo-quality">Quality: {(photo.quality_score * 100).toFixed(1)}%</p>
            )}
          </div>
        ))}
      </div>
      
      {error && <div className="error-message">{error}</div>}
      
      <div className="group-actions">
        <button
          onClick={handleDeduplicate}
          disabled={isLoading || selectedPhotoIds.size === 0}
          className="deduplicate-button"
        >
          {isLoading ? 'Deduplicating...' : `Deduplicate (${selectedPhotoIds.size})`}
        </button>
      </div>
    </div>
  );
}
