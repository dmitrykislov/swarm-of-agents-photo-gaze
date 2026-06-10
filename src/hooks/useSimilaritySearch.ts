/**
 * Debounced, paginated similarity search hook.
 * Re-fetches whenever threshold, page, jobId, or refreshKey changes.
 */
import React, { useEffect, useState, useRef } from 'react';
import { fetchSimilarityGroups } from '../api';
import { SimilarPhotosGroup } from '../components/SimilarPhotosGrid';

export interface UseSimilaritySearchResult {
  groups: SimilarPhotosGroup[];
  total: number;
  loading: boolean;
  error: string | null;
  setGroups: React.Dispatch<React.SetStateAction<SimilarPhotosGroup[]>>;
  /** Lets the grid keep the total/page count in sync when it removes a group
   * locally (after a deletion) without a full refetch. */
  setTotal: React.Dispatch<React.SetStateAction<number>>;
}

export function useSimilaritySearch(
  jobId: string,
  threshold: number,
  page: number = 0,
  pageSize: number = 20,
  debounceMs: number = 300,
  refreshKey: number = 0,
): UseSimilaritySearchResult {
  const [groups, setGroups] = useState<SimilarPhotosGroup[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    // Similarity groups are global (the backend ignores jobId), so always
    // load them — don't bail when there's no active job. jobId stays in the
    // dependency list only so a freshly-finished scan refreshes the list.

    // Cancel any in-flight request
    if (abortRef.current) abortRef.current.abort();
    if (timerRef.current) clearTimeout(timerRef.current);

    timerRef.current = setTimeout(async () => {
      const controller = new AbortController();
      abortRef.current = controller;
      setLoading(true);
      setError(null);
      try {
        const data = await fetchSimilarityGroups(threshold, page * pageSize, pageSize);
        if (!controller.signal.aborted) {
          setGroups(data.groups);
          setTotal(data.total);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : 'Unknown error');
          setGroups([]);
          setTotal(0);
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
  }, [jobId, threshold, page, pageSize, debounceMs, refreshKey]);

  return { groups, total, loading, error, setGroups, setTotal };
}
