import React, { useState, useMemo, useEffect, useCallback } from 'react';
import { deduplicatePhotos, fetchPhotoImageInfo, API_BASE_URL } from '../api';
import './GroupDetailView.css';

interface Photo {
  photo_id: number;
  filename: string;
  path: string;
  quality_score?: number;
  similarity_score?: number;
  file_size?: number;
  file_path?: string;
  mime_type?: string;
  uploaded_at?: string;
  width?: number;
  height?: number;
  created_date?: string;
}

interface SimilarityGroup {
  group_id: string;
  reference_photo: Photo;
  similar_photos: Photo[];
  best_reasons?: string[];
}

interface GroupDetailViewProps {
  group: SimilarityGroup;
  onClose: () => void;
  onDeleted?: (deletedIds: Set<number>) => void;
}

const API_BASE = API_BASE_URL;

function formatBytes(bytes?: number): string {
  if (!bytes) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

const GroupDetailView: React.FC<GroupDetailViewProps> = ({ group, onClose, onDeleted }) => {
  const allPhotos = useMemo(() => {
    return [group.reference_photo, ...group.similar_photos];
  }, [group]);

  const bestPhotoId = useMemo(() => group.reference_photo.photo_id, [group]);

  // Auto-select all NON-best photos for deletion by default
  const [selectedPhotoIds, setSelectedPhotoIds] = useState<Set<number>>(() => {
    const ids = new Set<number>();
    for (const p of [group.reference_photo, ...group.similar_photos]) {
      if (p.photo_id !== group.reference_photo.photo_id) ids.add(p.photo_id);
    }
    return ids;
  });
  const [deduplicating, setDeduplicating] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [lightboxPhotoId, setLightboxPhotoId] = useState<number | null>(null);
  const [lightboxLoading, setLightboxLoading] = useState(false);
  // Lazy image-info cache (width/height/created_date). Populated by the
  // lightbox when it opens — keeps these out of the slider's hot path.
  const [imageInfo, setImageInfo] = useState<Record<number, {
    width: number | null;
    height: number | null;
    created_date: string | null;
  }>>({});

  const [overrideBestId, setOverrideBestId] = useState<number | null>(null);

  const effectiveBestId = overrideBestId ?? bestPhotoId;

  const bestExplanation = overrideBestId
    ? ['Manually selected by you']
    : (group.best_reasons?.length ? group.best_reasons : ['First in similarity ranking']);

  // Compute similarity score range for the batch
  const similarityScores = group.similar_photos
    .map(p => p.similarity_score)
    .filter((s): s is number => s != null);
  const minSim = similarityScores.length > 0 ? Math.min(...similarityScores) : null;
  const maxSim = similarityScores.length > 0 ? Math.max(...similarityScores) : null;

  const handlePhotoToggle = (photoId: number) => {
    const next = new Set(selectedPhotoIds);
    if (next.has(photoId)) next.delete(photoId);
    else next.add(photoId);
    setSelectedPhotoIds(next);
  };

  const handleDeduplicate = async () => {
    if (selectedPhotoIds.size === 0) {
      setMessage('Select at least one photo to delete.');
      return;
    }
    setDeduplicating(true);
    setMessage(null);
    try {
      await deduplicatePhotos(Array.from(selectedPhotoIds));
      // Notify parent with the set of deleted IDs for local state update
      if (onDeleted) onDeleted(new Set(selectedPhotoIds));
    } catch (error) {
      setMessage(`Error: ${error instanceof Error ? error.message : 'Failed to deduplicate'}`);
    } finally {
      setDeduplicating(false);
    }
  };

  const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) onClose();
  };

  const currentLightboxPhoto = useMemo(
    () => allPhotos.find(p => p.photo_id === lightboxPhotoId) ?? null,
    [allPhotos, lightboxPhotoId]
  );

  // Lazy fetch image-info when the lightbox shows a new photo. Keeps the
  // slider's /similarity-groups hot path free of per-photo file I/O.
  useEffect(() => {
    if (lightboxPhotoId === null) return;
    if (imageInfo[lightboxPhotoId]) return;  // already cached
    let cancelled = false;
    fetchPhotoImageInfo(lightboxPhotoId)
      .then(info => {
        if (cancelled) return;
        setImageInfo(prev => ({
          ...prev,
          [info.photo_id]: {
            width: info.width,
            height: info.height,
            created_date: info.created_date,
          },
        }));
      })
      .catch(() => {/* image-info is decorative; ignore failures */});
    return () => { cancelled = true; };
  }, [lightboxPhotoId, imageInfo]);

  // Combine the base photo with its lazy-loaded image info for rendering.
  const lightboxPhotoWithInfo = useMemo(() => {
    if (!currentLightboxPhoto) return null;
    const info = imageInfo[currentLightboxPhoto.photo_id];
    return info
      ? { ...currentLightboxPhoto,
          width: info.width ?? currentLightboxPhoto.width,
          height: info.height ?? currentLightboxPhoto.height,
          created_date: info.created_date ?? currentLightboxPhoto.created_date }
      : currentLightboxPhoto;
  }, [currentLightboxPhoto, imageInfo]);

  const navigateLightbox = useCallback((dir: 1 | -1) => {
    if (lightboxPhotoId === null) return;
    const idx = allPhotos.findIndex(p => p.photo_id === lightboxPhotoId);
    if (idx < 0) return;
    const next = (idx + dir + allPhotos.length) % allPhotos.length;
    setLightboxPhotoId(allPhotos[next].photo_id);
  }, [lightboxPhotoId, allPhotos]);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      if (lightboxPhotoId !== null) setLightboxPhotoId(null);
      else onClose();
    } else if (lightboxPhotoId !== null && (e.key === 'ArrowRight' || e.key === 'ArrowDown')) {
      e.preventDefault();
      navigateLightbox(1);
    } else if (lightboxPhotoId !== null && (e.key === 'ArrowLeft' || e.key === 'ArrowUp')) {
      e.preventDefault();
      navigateLightbox(-1);
    }
  }, [lightboxPhotoId, onClose, navigateLightbox]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  return (
    <div className="group-detail-overlay" onClick={handleBackdropClick}>
      {lightboxPhotoId !== null && currentLightboxPhoto && (() => {
        const isBestLb = currentLightboxPhoto.photo_id === effectiveBestId;
        const isSelectedLb = selectedPhotoIds.has(currentLightboxPhoto.photo_id);
        return (
        <div className="lightbox-overlay" data-testid="lightbox" onClick={() => setLightboxPhotoId(null)}>
          {lightboxLoading && <div className="lightbox-spinner">Loading...</div>}
          <img
            src={`${API_BASE}/photos/${lightboxPhotoId}/full`}
            alt="Full resolution"
            className="lightbox-image"
            style={{ opacity: lightboxLoading ? 0.3 : 1 }}
            onClick={(e) => e.stopPropagation()}
            onLoadStart={() => setLightboxLoading(true)}
            onLoad={() => setLightboxLoading(false)}
            onError={() => setLightboxLoading(false)}
          />
          <button className="lightbox-close" onClick={() => setLightboxPhotoId(null)}>✕</button>
          <button className="lightbox-nav lightbox-prev" onClick={(e) => { e.stopPropagation(); setLightboxLoading(true); navigateLightbox(-1); }}>‹</button>
          <button className="lightbox-nav lightbox-next" onClick={(e) => { e.stopPropagation(); setLightboxLoading(true); navigateLightbox(1); }}>›</button>
          <div className="lightbox-info" onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
              <strong className="lightbox-info__filename">
                {currentLightboxPhoto.filename}
              </strong>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {/* Keep / delete toggle — also available for the
                    current Best photo, so the user can decide to
                    delete the auto-picked Best from full-screen too. */}
                {isSelectedLb ? (
                  <button
                    className="lb-badge lb-delete"
                    onClick={() => handlePhotoToggle(currentLightboxPhoto.photo_id)}
                    aria-label="Mark this photo as keep"
                  >
                    🗑 DELETING — click to keep
                  </button>
                ) : (
                  <button
                    className={`lb-badge ${isBestLb ? 'lb-keep' : 'lb-keep-btn'}`}
                    onClick={() => handlePhotoToggle(currentLightboxPhoto.photo_id)}
                    aria-label="Mark this photo for deletion"
                  >
                    {isBestLb ? '★ KEEPING (Best)' : '✓ KEEPING'} — click to delete
                  </button>
                )}
                {/* Mark-as-Best toggle — promotes this photo to the
                    cluster's reference. Hidden when already Best. */}
                {!isBestLb && (
                  <button
                    className="lb-badge lb-mark-best"
                    onClick={() => {
                      setOverrideBestId(currentLightboxPhoto.photo_id);
                      // If this photo was selected for deletion, un-mark it —
                      // the new Best is by definition kept.
                      if (isSelectedLb) handlePhotoToggle(currentLightboxPhoto.photo_id);
                    }}
                    aria-label="Mark this photo as the best of the group"
                  >
                    ★ Mark as Best
                  </button>
                )}
              </div>
            </div>
            <span className="lightbox-info__meta">
              {lightboxPhotoWithInfo!.width && lightboxPhotoWithInfo!.height && `${lightboxPhotoWithInfo!.width}×${lightboxPhotoWithInfo!.height}`}
              {currentLightboxPhoto.file_size && ` · ${formatBytes(currentLightboxPhoto.file_size)}`}
              {currentLightboxPhoto.mime_type && ` · ${currentLightboxPhoto.mime_type}`}
            </span>
            {lightboxPhotoWithInfo!.created_date && (
              <span className="lightbox-info__meta">
                Created: {new Date(lightboxPhotoWithInfo!.created_date as string).toLocaleString()}
              </span>
            )}
            {currentLightboxPhoto.file_path && (
              <span className="lightbox-info__path">
                {currentLightboxPhoto.file_path}
              </span>
            )}
            <span className="lightbox-info__hint">
              {allPhotos.findIndex(p => p.photo_id === lightboxPhotoId) + 1} / {allPhotos.length} · ← → to navigate · Esc to close
            </span>
          </div>
        </div>
        );
      })()}
      <div className="group-detail-modal">
        <div className="detail-header">
          <div>
            <h2 style={{ margin: 0 }}>Similar Photos &middot; {allPhotos.length} photos</h2>
            {minSim !== null && (
              <span style={{ fontSize: 13, color: '#888', fontWeight: 400 }}>
                Similarity: {minSim === maxSim
                  ? `${(minSim * 100).toFixed(1)}%`
                  : `${(minSim * 100).toFixed(1)}% – ${(maxSim! * 100).toFixed(1)}%`
                }
              </span>
            )}
          </div>
          <button className="close-button" onClick={onClose} aria-label="Close">✕</button>
        </div>

        <div className="best-explanation">
          <strong>★ Best photo kept:</strong> {allPhotos.find(p => p.photo_id === effectiveBestId)?.file_path ?? allPhotos.find(p => p.photo_id === effectiveBestId)?.filename ?? '?'}
          <ul>
            {bestExplanation.map((r, i) => <li key={i}>{r}</li>)}
          </ul>
        </div>

        <div className="detail-content">
          <div className="photos-grid">
            {allPhotos.map((photo) => {
              const isBest = photo.photo_id === effectiveBestId;
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
                    onClick={() => setLightboxPhotoId(photo.photo_id)}
                    onError={(e) => {
                      const img = e.currentTarget;
                      img.onerror = null;
                      img.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
                    }}
                    title="Click for full resolution"
                    style={{ cursor: 'zoom-in' }}
                  />
                  <div className="photo-metadata">
                    <p className="filename" title={photo.filename}><strong>{photo.filename}</strong></p>
                    <table className="meta-table">
                      <tbody>
                        {photo.similarity_score != null && (
                          <tr><td>Similarity</td><td>{(photo.similarity_score * 100).toFixed(1)}%</td></tr>
                        )}
                        {photo.width && photo.height && (
                          <tr><td>Resolution</td><td>{photo.width} x {photo.height}</td></tr>
                        )}
                        <tr><td>File size</td><td>{formatBytes(photo.file_size)}</td></tr>
                        {photo.mime_type && <tr><td>Type</td><td>{photo.mime_type}</td></tr>}
                        {photo.created_date && (
                          <tr><td>Created</td><td>{new Date(photo.created_date).toLocaleString()}</td></tr>
                        )}
                        {photo.file_path && (
                          <tr><td>Path</td><td className="path-cell" title={photo.file_path}>{photo.file_path.split('/').pop()}</td></tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                  <div className="card-actions">
                    {isBest && !isSelected ? (
                      <div className="fate-badge fate-keep">★ KEEPING — Best photo</div>
                    ) : isSelected ? (
                      <div className="fate-badge fate-delete" onClick={() => handlePhotoToggle(photo.photo_id)}>
                        🗑 DELETING — <span className="fate-toggle-hint">click to keep</span>
                      </div>
                    ) : (
                      <div className="fate-badge fate-keep-manual" onClick={() => handlePhotoToggle(photo.photo_id)}>
                        ✓ KEEPING — <span className="fate-toggle-hint">click to delete</span>
                      </div>
                    )}
                    {!isBest && (
                      <button
                        className="mark-best-btn"
                        onClick={() => {
                          setOverrideBestId(photo.photo_id);
                          const next = new Set(allPhotos.map(p => p.photo_id));
                          next.delete(photo.photo_id);
                          setSelectedPhotoIds(next);
                        }}
                        title="Override automatic selection and keep this photo instead"
                      >
                        Mark as Best
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="detail-footer">
          {message && (
            <p className={`message ${message.includes('Error') ? 'error' : 'success'}`}>
              {message}
            </p>
          )}
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
            <button
              className="deduplicate-button"
              onClick={handleDeduplicate}
              disabled={selectedPhotoIds.size === 0 || deduplicating}
            >
              {deduplicating ? 'Deleting...' : `Delete ${selectedPhotoIds.size} photo(s)`}
            </button>
            <button
              className="select-all-btn"
              onClick={() => {
                const allIds = new Set(allPhotos.map(p => p.photo_id));
                setSelectedPhotoIds(allIds);
              }}
              title="Select every photo in this group for deletion, including the best"
            >
              Select all (including best)
            </button>
            <button
              className="select-all-btn"
              onClick={() => {
                setSelectedPhotoIds(new Set());
              }}
            >
              Deselect all
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GroupDetailView;
