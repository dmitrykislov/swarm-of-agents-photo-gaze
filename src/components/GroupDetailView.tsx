import React, { useState } from 'react';
import { deduplicatePhotos } from '../api';
import './GroupDetailView.css';

interface Photo {
  photo_id: number;
  filename: string;
  path: string;
  quality_score?: number;
  resolution?: string;
  file_size?: string;
  created_date?: string;
  similarity_score?: number;
}

interface SimilarPhotosGroup {
  group_id: string;
  reference_photo: Photo;
  similar_photos: Photo[];
}

interface GroupDetailViewProps {
  group: SimilarPhotosGroup;
  onClose: () => void;
}

const GroupDetailView: React.FC<GroupDetailViewProps> = ({ group, onClose }) => {
  const [selectedPhotoIds, setSelectedPhotoIds] = useState<Set<number>>(new Set());
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const allPhotos: Photo[] = [group.reference_photo, ...group.similar_photos];

  // Determine best photo: highest quality_score wins
  const bestPhotoId = allPhotos.reduce((bestId, photo) => {
    const bestPhoto = allPhotos.find((p) => p.photo_id === bestId);
    const bestScore = bestPhoto?.quality_score ?? 0;
    const currentScore = photo.quality_score ?? 0;
    return currentScore > bestScore ? photo.photo_id : bestId;
  }, allPhotos[0].photo_id);

  const toggleSelection = (photoId: number) => {
    setSelectedPhotoIds((prev) => {
      const next = new Set(prev);
      if (next.has(photoId)) {
        next.delete(photoId);
      } else {
        next.add(photoId);
      }
      return next;
    });
  };

  const handleDelete = async () => {
    const photoIds = Array.from(selectedPhotoIds);
    if (photoIds.length === 0) return;
    try {
      const result = await deduplicatePhotos(photoIds);
      setStatusMessage(`Successfully deleted ${result.deleted} photo(s)`);
      setSelectedPhotoIds(new Set());
    } catch (err: any) {
      setStatusMessage(`Error: ${err.message}`);
    }
  };

  const handleOverlayClick = (e: React.MouseEvent<HTMLDivElement>) => {
    // Close when clicking the overlay backdrop, not the modal content
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div className="group-detail-overlay" onClick={handleOverlayClick}>
      <div className="group-detail-modal" onClick={(e) => e.stopPropagation()}>
        <div className="group-detail-header">
          <h2>Group Detail</h2>
          <button className="close-button" aria-label="Close" onClick={onClose}>
            &times;
          </button>
        </div>

        {statusMessage && (
          <div className="status-message">{statusMessage}</div>
        )}

        <div className="group-detail-photos">
          {allPhotos.map((photo) => {
            const isBest = photo.photo_id === bestPhotoId;
            return (
              <div
                key={photo.photo_id}
                className={`photo-card${isBest ? ' best-photo' : ''}`}
              >
                {isBest && <span className="best-indicator">★ Best</span>}
                <img
                  src={photo.path}
                  alt={photo.filename}
                  className="detail-image"
                />
                <div className="photo-metadata">
                  <p className="photo-filename">{photo.filename}</p>
                  <p>Quality: {((photo.quality_score ?? 0) * 100).toFixed(1)}%</p>
                  {photo.resolution && <p>Resolution: {photo.resolution}</p>}
                  {photo.file_size && <p>Size: {photo.file_size}</p>}
                  {photo.created_date && <p>Created: {photo.created_date}</p>}
                  <p>Path: {photo.path}</p>
                  {photo.similarity_score !== undefined && (
                    <p>Similarity: {(photo.similarity_score * 100).toFixed(1)}%</p>
                  )}
                </div>
                <div className="photo-checkbox">
                  <input
                    type="checkbox"
                    checked={selectedPhotoIds.has(photo.photo_id)}
                    disabled={isBest}
                    title={isBest ? 'Cannot delete the best photo' : 'Select for deletion'}
                    onChange={() => toggleSelection(photo.photo_id)}
                  />
                </div>
              </div>
            );
          })}
        </div>

        <div className="group-detail-actions">
          <button
            className="delete-button"
            onClick={handleDelete}
            disabled={selectedPhotoIds.size === 0}
          >
            Delete Selected ({selectedPhotoIds.size})
          </button>
        </div>
      </div>
    </div>
  );
};

export default GroupDetailView;
