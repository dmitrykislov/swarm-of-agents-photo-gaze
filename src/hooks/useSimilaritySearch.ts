/**
 * Debounced similarity search hook.
 * Re-fetches whenever threshold, jobId, or refreshKey changes.
 */
import React, { useEffect, useState, useRef } from 'react';
import { searchSimilarPhotos } from '../api';
import { SimilarPhotosGroup } from '../components/SimilarPhotosGrid';

export interface UseSimilaritySearchResult {
  groups: SimilarPhotosGroup[];
  loading: boolean;
  error: string | null;
  setGroups: React.Dispatch<React.SetStateAction<SimilarPhotosGroup[]>>;
}

export function useSimilaritySearch(
  jobId: string,
  threshold: number,
  debounceMs: number = 300,
  refreshKey: number = 0
): UseSimilaritySearchResult {
  const [groups, setGroups] = useState<SimilarPhotosGroup[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    // Similarity groups are global (the backend's /similarity-groups ignores
    // jobId), so always load them — don't bail when there's no active job.
    // jobId stays in the dependency list only so a freshly-finished scan
    // refreshes the list.

    // Cancel any in-flight request
    if (abortRef.current) abortRef.current.abort();
    if (timerRef.current) clearTimeout(timerRef.current);

    timerRef.current = setTimeout(async () => {
      const controller = new AbortController();
      abortRef.current = controller;
      setLoading(true);
      setError(null);
      try {
        const data = await searchSimilarPhotos(jobId, threshold);
        if (!controller.signal.aborted) {
          setGroups(data);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : 'Unknown error');
          setGroups([]);
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }, debounceMs);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (abortRef.current) abortRef.current.abort();
    };
  }, [jobId, threshold, debounceMs, refreshKey]);

  return { groups, loading, error, setGroups };
}
