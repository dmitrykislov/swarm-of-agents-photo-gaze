import React, { useState, useCallback, useEffect } from 'react';
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

const PAGE_SIZE = 20;

interface SimilarPhotosGridProps {
  jobId: string;
  threshold?: number;
  /** Bumped by the parent to force a refresh (e.g. after a folder is removed). */
  refreshSignal?: number;
}

const SimilarPhotosGrid: React.FC<SimilarPhotosGridProps> = ({ jobId, threshold = 0.5, refreshSignal = 0 }) => {
  const [detailGroup, setDetailGroup] = useState<SimilarPhotosGroup | null>(null);
  const [page, setPage] = useState(0);

  const { groups, total, loading, error, setGroups, setTotal } = useSimilaritySearch(
    jobId, threshold, page, PAGE_SIZE, 300, refreshSignal,
  );

  // Neutral placeholder for thumbnails that fail to load (e.g. a file moved
  // out from under the index) — avoids the browser's broken-image icon.
  const onImgError = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    img.onerror = null;
    img.classList.remove('loading');
    img.classList.add('thumb-broken');
    img.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
  };

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  // Reset to the first page whenever the threshold or an external refresh
  // changes the result set, so we never sit on an out-of-range page.
  useEffect(() => { setPage(0); }, [threshold, refreshSignal]);

  // Clamp if the last page shrank (e.g. deletions reduced the total).
  useEffect(() => {
    if (page > totalPages - 1) setPage(totalPages - 1);
  }, [page, totalPages]);

  /** After photos are deleted from a group, update local state instead of a
   * full reload. A group that drops to <2 photos is removed entirely; we
   * decrement `total` so the header count and page count stay in sync (the
   * server dropped the same group, so this matches a refetch). */
  const handleDeletedFromGroup = useCallback((groupId: string, deletedIds: Set<number>) => {
    let dropped = 0;
    setGroups(prev => {
      const updated: SimilarPhotosGroup[] = [];
      for (const g of prev) {
        if (g.group_id !== groupId) {
          updated.push(g);
          continue;
        }
        const allPhotos = [g.reference_photo, ...g.similar_photos]
          .filter(p => !deletedIds.has(p.photo_id));
        if (allPhotos.length <= 1) { dropped += 1; continue; }
        updated.push({
          ...g,
          reference_photo: allPhotos[0],
          similar_photos: allPhotos.slice(1),
        });
      }
      return updated;
    });
    if (dropped > 0) setTotal((t) => Math.max(0, t - dropped));
    setDetailGroup(null);
  }, [setGroups, setTotal]);

  const getQualityLabel = (score: number): string => {
    if (score >= 0.85) return 'Excellent';
    if (score >= 0.7) return 'Good';
    if (score >= 0.5) return 'Fair';
    return 'Poor';
  };

  // NOTE: similarity groups are GLOBAL — they come from the whole embedding
  // index, not a specific job. So we do NOT gate the view on an active jobId
  // (which is empty after a page reload); duplicates stay visible whenever the
  // index has them. The empty-state below covers "index has no groups yet".

  if (error) {
    return (
      <div className="similar-photos-container">
        <p className="error">Error: {error}</p>
      </div>
    );
  }

  // First load (no data yet): show the loader. On page changes we keep the
  // previous page visible (groups stays populated) so the view doesn't flash.
  if (groups.length === 0) {
    return (
      <div className="similar-photos-container">
        <p className="muted">{loading ? 'Loading similar photos…' : 'No similar photos found.'}</p>
      </div>
    );
  }

  const pager = totalPages > 1 ? (
    <div className="pager">
      <button className="btn btn--sm" disabled={page === 0} onClick={() => setPage(0)}>« First</button>
      <button className="btn btn--sm" disabled={page === 0} onClick={() => setPage((p) => Math.max(0, p - 1))}>‹ Prev</button>
      <span className="pager__info">Page {page + 1} of {totalPages}</span>
      <button className="btn btn--sm" disabled={page >= totalPages - 1} onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}>Next ›</button>
      <button className="btn btn--sm" disabled={page >= totalPages - 1} onClick={() => setPage(totalPages - 1)}>Last »</button>
    </div>
  ) : null;

  const rangeStart = page * PAGE_SIZE + 1;
  const rangeEnd = page * PAGE_SIZE + groups.length;

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
      <div className="grid-head">
        <h2 className="grid-title">
          Similar groups <span className="grid-count">{total.toLocaleString()}</span>
          {total > PAGE_SIZE && (
            <span className="grid-range"> · showing {rangeStart.toLocaleString()}–{rangeEnd.toLocaleString()}</span>
          )}
        </h2>
        {pager}
      </div>
      {/* Results area: while a threshold/page fetch is in flight, dim the
          current results and show an overlay so the slider feels responsive
          even when the backend is still clustering. */}
      <div style={{ position: 'relative' }}>
        {loading && (
          <div
            aria-busy="true"
            style={{
              position: 'absolute',
              inset: 0,
              zIndex: 5,
              display: 'flex',
              alignItems: 'flex-start',
              justifyContent: 'center',
              paddingTop: '2rem',
              background: 'rgba(255,255,255,0.55)',
              pointerEvents: 'none',
            }}
          >
            <span className="muted">Updating results…</span>
          </div>
        )}
        <div
          style={{
            opacity: loading ? 0.4 : 1,
            transition: 'opacity 120ms ease',
            pointerEvents: loading ? 'none' : 'auto',
          }}
        >
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
                onError={onImgError}
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
                  onError={onImgError}
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
      </div>
      {totalPages > 1 && <div className="pager pager--bottom">{pager}</div>}
    </div>
  );
};
/* (bottom pager reuses the .pager element; .pager--bottom only adds spacing) */

export default SimilarPhotosGrid;
