import React, { useState, useMemo } from 'react';
import { deduplicatePhotos } from '../api';
import './GroupDetailView.css';

interface Photo {
  photo_id: number;
  filename: string;
  path: string;
  quality_score: number;
  resolution?: string;
  file_size?: string;
  created_date?: string;
  similarity_score?: number;
}

interface SimilarityGroup {
  group_id: string;
  reference_photo: Photo;
  similar_photos: Photo[];
}

interface GroupDetailViewProps {
  group: SimilarityGroup;
  onClose: () => void;
}

/**
 * Full-screen detail view for a similarity group.
 * Displays all images at larger size with metadata, best-photo indicator,
 * checkbox selection, and deduplicate button.
 */
const GroupDetailView: React.FC<GroupDetailViewProps> = ({ group, onClose }) => {
  const [selectedPhotoIds, setSelectedPhotoIds] = useState<Set<number>>(new Set());
  const [deduplicating, setDeduplicating] = useState(false);
  const [deduplicateMessage, setDeduplicateMessage] = useState<string | null>(null);

  // Combine reference photo and similar photos into one list
  const allPhotos = useMemo(() => {
    return [group.reference_photo, ...group.similar_photos];
  }, [group]);

  // Determine best photo: highest quality_score, then earliest created_date
  const bestPhotoId = useMemo(() => {
    if (allPhotos.length === 0) return null;
    return allPhotos.reduce((best, current) => {
      if (current.quality_score > best.quality_score) return current;
      if (current.quality_score === best.quality_score) {
        const bestDate = best.created_date ? new Date(best.created_date).getTime() : Infinity;
        const currentDate = current.created_date ? new Date(current.created_date).getTime() : Infinity;
        if (currentDate < bestDate) return current;
      }
      return best;
    }).photo_id;
  }, [allPhotos]);

  const handlePhotoToggle = (photoId: number) => {
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
      setDeduplicateMessage('Please select at least one photo to delete.');
      return;
    }
    setDeduplicating(true);
    setDeduplicateMessage(null);
    try {
      const result = await deduplicatePhotos(Array.from(selectedPhotoIds));
      setDeduplicateMessage(`Successfully deleted ${result.deleted} photo(s).`);
      setSelectedPhotoIds(new Set());
    } catch (error) {
      setDeduplicateMessage(
        `Error: ${error instanceof Error ? error.message : 'Failed to deduplicate'}`
      );
    } finally {
      setDeduplicating(false);
    }
  };

  const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div className="group-detail-overlay" onClick={handleBackdropClick}>
      <div className="group-detail-modal">
        <div className="detail-header">
          <h2>Group Detail</h2>
          <button className="close-button" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        <div className="detail-content">
          <div className="photos-grid">
            {allPhotos.map((photo) => {
              const isBest = photo.photo_id === bestPhotoId;
              const isSelected = selectedPhotoIds.has(photo.photo_id);
              return (
                <div
                  key={photo.photo_id}
                  className={`photo-card ${isBest ? 'best-photo' : ''} ${isSelected ? 'selected' : ''}`}
                >
                  {isBest && <div className="best-indicator">★ Best</div>}
                  <img
                    src={photo.path}
                    alt={photo.filename}
                    className="detail-image"
                  />
                  <div className="photo-metadata">
                    <p className="filename"><strong>{photo.filename}</strong></p>
                    {photo.resolution && <p>Resolution: {photo.resolution}</p>}
                    {photo.file_size && <p>Size: {photo.file_size}</p>}
                    {photo.created_date && <p>Created: {photo.created_date}</p>}
                    <p className="file-path">Path: {photo.path}</p>
                    <p className="quality-score">Quality: {(photo.quality_score * 100).toFixed(1)}%</p>
                  </div>
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => handlePhotoToggle(photo.photo_id)}
                      disabled={isBest}
                      title={isBest ? 'Cannot delete the best photo' : ''}
                    />
                    {isBest ? 'Best (cannot delete)' : 'Mark for deletion'}
                  </label>
                </div>
              );
            })}
          </div>
        </div>

        <div className="detail-footer">
          {deduplicateMessage && (
            <p className={`message ${deduplicateMessage.includes('Error') ? 'error' : 'success'}`}>
              {deduplicateMessage}
            </p>
          )}
          <button
            className="deduplicate-button"
            onClick={handleDeduplicate}
            disabled={selectedPhotoIds.size === 0 || deduplicating}
          >
            {deduplicating ? 'Deleting...' : `Delete Selected (${selectedPhotoIds.size})`}
          </button>
        </div>
      </div>
    </div>
  );
};

export default GroupDetailView;
