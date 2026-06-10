import React, { useState, useCallback } from 'react';
import { useSimilaritySearch } from '../hooks/useSimilaritySearch';
import GroupDetailView from './GroupDetailView';
import './SimilarPhotosGrid.css';

export interface Photo {
  photo_id: number;
  filename: string;
  path: string;
  quality_score?: number;
  similarity_score?: number;
}

export interface SimilarPhotosGroup {
  group_id: string;
  reference_photo: Photo;
  similar_photos: Photo[];
  best_reasons?: string[];
}

interface SimilarPhotosGridProps {
  jobId: string;
  threshold?: number;
}

const SimilarPhotosGrid: React.FC<SimilarPhotosGridProps> = ({ jobId, threshold = 0.5 }) => {
  const [detailGroup, setDetailGroup] = useState<SimilarPhotosGroup | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const { groups, loading, error, setGroups } = useSimilaritySearch(jobId, threshold, 300, refreshKey);

  /** After photos are deleted from a group, update local state instead of full reload */
  const handleDeletedFromGroup = useCallback((groupId: string, deletedIds: Set<number>) => {
    setGroups(prev => {
      const updated: SimilarPhotosGroup[] = [];
      for (const g of prev) {
        if (g.group_id !== groupId) {
          updated.push(g);
          continue;
        }
        // Filter out deleted photos from both reference and similar
        const allPhotos = [g.reference_photo, ...g.similar_photos]
          .filter(p => !deletedIds.has(p.photo_id));
        // If 0 or 1 photos remain, remove the entire group row
        if (allPhotos.length <= 1) continue;
        // Rebuild group: first photo becomes reference, rest become similar
        updated.push({
          ...g,
          reference_photo: allPhotos[0],
          similar_photos: allPhotos.slice(1),
        });
      }
      return updated;
    });
    setDetailGroup(null);
  }, [setGroups]);

  const getQualityLabel = (score: number): string => {
    if (score >= 0.85) return 'Excellent';
    if (score >= 0.7) return 'Good';
    if (score >= 0.5) return 'Fair';
    return 'Poor';
  };

  const getQualityClass = (score: number): string => {
    if (score >= 0.85) return 'quality-excellent';
    if (score >= 0.7) return 'quality-good';
    if (score >= 0.5) return 'quality-fair';
    return 'quality-poor';
  };

  // NOTE: similarity groups are GLOBAL — they come from the whole embedding
  // index, not a specific job. So we do NOT gate the view on an active jobId
  // (which is empty after a page reload); duplicates stay visible whenever the
  // index has them. The empty-state below covers "index has no groups yet".

  if (loading) {
    return (
      <div className="similar-photos-container">
        <p>Loading similar photos...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="similar-photos-container">
        <p className="error">Error: {error}</p>
      </div>
    );
  }

  if (groups.length === 0) {
    return (
      <div className="similar-photos-container">
        <p>No similar photos found.</p>
      </div>
    );
  }

  return (
    <div className="similar-photos-container">
      {detailGroup && (
        <GroupDetailView
          group={detailGroup}
          onClose={() => setDetailGroup(null)}
          onDeleted={(deletedIds: Set<number>) => {
            handleDeletedFromGroup(detailGroup.group_id, deletedIds);
          }}
        />
      )}
      <h2 className="grid-title">Similar Photos ({groups.length} groups)</h2>
      {groups.map((group) => (
        <div
          key={group.group_id}
          className="group-container"
          onClick={() => setDetailGroup(group)}
          style={{ cursor: 'pointer' }}
          title="Click to inspect and deduplicate this group"
        >
          <div className="group-header">
            <span>★ {group.reference_photo.filename}</span>
            <span className="match-count">{group.similar_photos.length} similar</span>
          </div>
          <div className="photos-grid">
            <div className="photo-card reference">
              <img
                src={group.reference_photo.path}
                alt={group.reference_photo.filename}
                className="thumbnail loading"
                onLoad={(e) => e.currentTarget.classList.remove('loading')}
              />
              <div className="photo-info">
                <div className="photo-filename">{group.reference_photo.filename}</div>
                {group.reference_photo.quality_score !== undefined && (
                  <div className="photo-score">
                    Quality: <strong>{getQualityLabel(group.reference_photo.quality_score)} ({(group.reference_photo.quality_score * 100).toFixed(1)}%)</strong>
                  </div>
                )}
              </div>
            </div>
            {group.similar_photos.map((photo) => (
              <div key={photo.photo_id} className="photo-card">
                <img
                  src={photo.path}
                  alt={photo.filename}
                  className="thumbnail loading"
                  onLoad={(e) => e.currentTarget.classList.remove('loading')}
                />
                <div className="photo-info">
                  <div className="photo-filename">{photo.filename}</div>
                  {photo.similarity_score !== undefined && (
                    <div className="photo-score">
                      Similarity: <strong>{(photo.similarity_score * 100).toFixed(1)}%</strong>
                    </div>
                  )}
                  {photo.quality_score !== undefined && (
                    <div className="quality-badge">
                      Quality: {getQualityLabel(photo.quality_score)} ({(photo.quality_score * 100).toFixed(1)}%)
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

export default SimilarPhotosGrid;
