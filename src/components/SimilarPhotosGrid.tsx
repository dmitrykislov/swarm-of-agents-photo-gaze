import React, { useState } from 'react';
import { useSimilaritySearch } from '../hooks/useSimilaritySearch';
import GroupDetailView from './GroupDetailView';
import './SimilarPhotosGrid.css';

interface Photo {
  photo_id: number;
  filename: string;
  path: string;
  quality_score?: number;
  similarity_score?: number;
  resolution?: string;
  file_size?: string;
  created_date?: string;
}

interface SimilarityGroup {
  group_id: string;
  reference_photo: Photo;
  similar_photos: Photo[];
}

interface SimilarPhotosGridProps {
  jobId: string;
  threshold?: number;
}

function getQualityLabel(score: number): string {
  if (score >= 0.9) return 'Excellent';
  if (score >= 0.7) return 'Good';
  if (score >= 0.5) return 'Fair';
  return 'Poor';
}

const SimilarPhotosGrid: React.FC<SimilarPhotosGridProps> = ({ jobId, threshold }) => {
  const { groups, loading, error } = useSimilaritySearch(jobId, threshold);
  const [selectedGroup, setSelectedGroup] = useState<SimilarityGroup | null>(null);

  if (!jobId) {
    return <p>No job selected. Process a job to view similar photos.</p>;
  }

  if (loading) {
    return <p>Loading similar photos...</p>;
  }

  if (error) {
    return <p>Error: {error}</p>;
  }

  if (!groups || groups.length === 0) {
    return <p>No similar photos found.</p>;
  }

  return (
    <div className="similar-photos-grid">
      <h2>Similar Photos ({groups.length} groups)</h2>
      {groups.map((group: SimilarityGroup) => (
        <div
          key={group.group_id}
          className="group-container"
          role="button"
          tabIndex={0}
          onClick={() => setSelectedGroup(group)}
          onKeyDown={(e) => { if (e.key === 'Enter') setSelectedGroup(group); }}
        >
          <div className="group-header">
            <span>{group.similar_photos.length} matches</span>
          </div>
          <div className="photo-row">
            {/* Reference photo */}
            <div className="photo-item reference">
              <img
                className="thumbnail"
                src={`/api/photos/${group.reference_photo.photo_id}/thumbnail`}
                alt={group.reference_photo.filename}
                loading="lazy"
              />
              <p>{group.reference_photo.filename}</p>
              {group.reference_photo.quality_score != null && (
                <p>
                  <span>Quality:</span>{' '}
                  {getQualityLabel(group.reference_photo.quality_score)} ({(group.reference_photo.quality_score * 100).toFixed(1)}%)
                </p>
              )}
            </div>
            {/* Similar photos */}
            {group.similar_photos.map((photo: Photo) => (
              <div key={photo.photo_id} className="photo-item similar">
                <img
                  className="thumbnail"
                  src={`/api/photos/${photo.photo_id}/thumbnail`}
                  alt={photo.filename}
                  loading="lazy"
                />
                <p>{photo.filename}</p>
                {photo.similarity_score != null && (
                  <p>{(photo.similarity_score * 100).toFixed(1)}%</p>
                )}
                {photo.quality_score != null && (
                  <p>
                    <span>Quality:</span>{' '}
                    {getQualityLabel(photo.quality_score)} ({(photo.quality_score * 100).toFixed(1)}%)
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      ))}

      {/* Full-screen overlay for group detail */}
      {selectedGroup && (
        <GroupDetailView
          group={selectedGroup}
          onClose={() => setSelectedGroup(null)}
        />
      )}
    </div>
  );
};

export default SimilarPhotosGrid;
