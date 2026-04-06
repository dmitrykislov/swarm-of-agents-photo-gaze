import React, { useState, useMemo, useCallback } from 'react';
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
 * Full-screen overlay showing all images in a similarity group at larger size,
 * with metadata, best-photo indicator, and deduplication controls.
 */
const GroupDetailView: React.FC<GroupDetailViewProps> = ({ group, onClose }) => {
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [statusMessage, setStatusMessage] = useState<string>('');

  // Collect all photos: reference + similar
  const allPhotos = useMemo(() => {
    return [group.reference_photo, ...group.similar_photos];
  }, [group]);

  // Determine best photo: highest quality_score (O(n) single pass)
  const bestPhotoId = useMemo(() => {
    let bestId = allPhotos[0]?.photo_id;
    let bestScore = allPhotos[0]?.quality_score ?? 0;
    for (let i = 1; i < allPhotos.length; i++) {
      const score = allPhotos[i].quality_score ?? 0;
      if (score > bestScore) {
        bestScore = score;
        bestId = allPhotos[i].photo_id;
      }
    }
    return bestId;
  }, [allPhotos]);

  const toggleSelection = useCallback((photoId: number) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(photoId)) {
        next.delete(photoId);
      } else {
        next.add(photoId);
      }
      return next;
    });
  }, []);

  const handleDelete = async () => {
    if (selectedIds.size === 0) return;
    try {
      const ids = Array.from(selectedIds);
      const result = await deduplicatePhotos(ids);
      setStatusMessage(`Successfully deleted ${result.deleted} photo(s).`);
      setSelectedIds(new Set());
    } catch (err: any) {
      setStatusMessage(`Error: ${err.message}`);
    }
  };

  /** Close when clicking the overlay backdrop (not the modal content) */
  const handleOverlayClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div className="group-detail-overlay" onClick={handleOverlayClick} data-testid="group-detail-overlay">
      <div className="group-detail-modal">
        <div className="modal-header">
          <h2>Group Detail</h2>
          <button className="close-button" onClick={onClose} aria-label="Close">✕</button>
        </div>

        <div className="photos-grid">
          {allPhotos.map(photo => {
            const isBest = photo.photo_id === bestPhotoId;
            const isSelected = selectedIds.has(photo.photo_id);
            return (
              <div
                key={photo.photo_id}
                className={`photo-card${isBest ? ' best-photo' : ''}`}
              >
                {isBest && <span className="best-indicator">★ Best</span>}
                <img
                  className="detail-image"
                  src={`/api/photos/${photo.photo_id}/image`}
                  alt={photo.filename}
                />
                <div className="photo-metadata">
                  <p className="filename">{photo.filename}</p>
                  {photo.resolution && <p>Resolution: {photo.resolution}</p>}
                  {photo.file_size && <p>Size: {photo.file_size}</p>}
                  {photo.created_date && <p>Created: {photo.created_date}</p>}
                  <p>Path: {photo.path}</p>
                  {photo.quality_score != null && (
                    <p>Quality: {(photo.quality_score * 100).toFixed(1)}%</p>
                  )}
                </div>
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={isSelected}
                    disabled={isBest}
                    title={isBest ? 'Cannot delete the best photo' : 'Select for deletion'}
                    onChange={() => toggleSelection(photo.photo_id)}
                  />
                  {isBest ? 'Keep (best)' : 'Select'}
                </label>
              </div>
            );
          })}
        </div>

        <div className="modal-footer">
          <button
            className="delete-button"
            disabled={selectedIds.size === 0}
            onClick={handleDelete}
          >
            Delete Selected ({selectedIds.size})
          </button>
          {statusMessage && <p className="status-message">{statusMessage}</p>}
        </div>
      </div>
    </div>
  );
};

export default GroupDetailView;
