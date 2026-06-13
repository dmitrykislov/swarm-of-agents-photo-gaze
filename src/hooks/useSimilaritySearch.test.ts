/**
 * Unit tests for useSimilaritySearch hook.
 * Tests debounced search triggering and threshold change handling.
 */
import { renderHook, waitFor } from '@testing-library/react';
import { useSimilaritySearch } from './useSimilaritySearch';
import * as api from '../api';

jest.mock('../api');

const mockGroups = [
  {
    group_id: 'g1',
    reference_photo: { photo_id: 1, filename: 'a.jpg', path: '/t/1' },
    similar_photos: [{ photo_id: 2, filename: 'b.jpg', path: '/t/2', similarity_score: 0.95 }],
  },
];

beforeEach(() => {
  jest.clearAllMocks();
  (api.fetchSimilarityGroups as jest.Mock).mockResolvedValue({ groups: mockGroups, total: 1 });
});

// debounceMs=0 so the debounced fetch fires on the next tick (no fake timers).
const ARGS = ['job', 0.9, 0, 20, 0] as const;

describe('useSimilaritySearch', () => {
  it('fetches groups on mount with (threshold, skip, limit)', async () => {
    const { result } = renderHook(() => useSimilaritySearch(...ARGS));
    await waitFor(() => expect(result.current.groups).toHaveLength(1));
    expect(api.fetchSimilarityGroups).toHaveBeenCalledWith(0.9, 0, 20);
    expect(result.current.total).toBe(1);
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it('refetches when the threshold changes', async () => {
    const { rerender } = renderHook(
      ({ t }) => useSimilaritySearch('job', t, 0, 20, 0),
      { initialProps: { t: 0.9 } },
    );
    await waitFor(() => expect(api.fetchSimilarityGroups).toHaveBeenCalledWith(0.9, 0, 20));
    rerender({ t: 0.7 });
    await waitFor(() => expect(api.fetchSimilarityGroups).toHaveBeenCalledWith(0.7, 0, 20));
  });

  it('requests the right page slice when page changes', async () => {
    const { rerender } = renderHook(
      ({ p }) => useSimilaritySearch('job', 0.9, p, 20, 0),
      { initialProps: { p: 0 } },
    );
    await waitFor(() => expect(api.fetchSimilarityGroups).toHaveBeenCalledWith(0.9, 0, 20));
    rerender({ p: 2 });
    await waitFor(() => expect(api.fetchSimilarityGroups).toHaveBeenCalledWith(0.9, 40, 20));
  });

  it('surfaces an error and clears results on failure', async () => {
    (api.fetchSimilarityGroups as jest.Mock).mockRejectedValueOnce(new Error('boom'));
    const { result } = renderHook(() => useSimilaritySearch(...ARGS));
    await waitFor(() => expect(result.current.error).toBe('boom'));
    expect(result.current.groups).toEqual([]);
    expect(result.current.total).toBe(0);
  });

  it('clears a prior error after a later success', async () => {
    (api.fetchSimilarityGroups as jest.Mock).mockRejectedValueOnce(new Error('boom'));
    const { result, rerender } = renderHook(
      ({ t }) => useSimilaritySearch('job', t, 0, 20, 0),
      { initialProps: { t: 0.9 } },
    );
    await waitFor(() => expect(result.current.error).toBe('boom'));
    rerender({ t: 0.8 });
    await waitFor(() => expect(result.current.error).toBeNull());
    expect(result.current.groups).toHaveLength(1);
  });
});
