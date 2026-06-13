import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import SimilarPhotosGrid from './SimilarPhotosGrid';
import * as api from '../api';

jest.mock('../api');

// Current group shape (built by the backend's _build_group_for_component).
const mockGroups = [
  {
    group_id: 'group_1',
    similarity_score: 0.9,
    quality_score: 0.95,
    reference_photo: {
      photo_id: 1, filename: 'reference1.jpg', path: '/t/1',
      quality_score: 0.95, similarity_score: 1.0,
    },
    similar_photos: [
      { photo_id: 2, filename: 'similar1.jpg', path: '/t/2', similarity_score: 0.92, quality_score: 0.88 },
      { photo_id: 3, filename: 'similar2.jpg', path: '/t/3', similarity_score: 0.85, quality_score: 0.75 },
    ],
    best_reasons: ['Largest file'],
  },
  {
    group_id: 'group_2',
    similarity_score: 0.8,
    quality_score: 0.82,
    reference_photo: {
      photo_id: 4, filename: 'reference2.jpg', path: '/t/4',
      quality_score: 0.82, similarity_score: 1.0,
    },
    similar_photos: [
      { photo_id: 5, filename: 'similar3.jpg', path: '/t/5', similarity_score: 0.78, quality_score: 0.70 },
    ],
    best_reasons: ['First in ranking'],
  },
];

describe('SimilarPhotosGrid', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (api.fetchSimilarityGroups as jest.Mock).mockResolvedValue({ groups: mockGroups, total: 2 });
  });

  it('renders the group count and per-group headers', async () => {
    render(<SimilarPhotosGrid jobId="job" threshold={0.9} />);
    await waitFor(() => expect(screen.getByText('Similar groups')).toBeInTheDocument());
    // match-count badges: "<n> similar"
    expect(screen.getByText('2 similar')).toBeInTheDocument();
    expect(screen.getByText('1 similar')).toBeInTheDocument();
  });

  it('renders reference and similar photo filenames', async () => {
    render(<SimilarPhotosGrid jobId="job" threshold={0.9} />);
    await waitFor(() => expect(screen.getByText('reference1.jpg')).toBeInTheDocument());
    expect(screen.getByText('reference2.jpg')).toBeInTheDocument();
    expect(screen.getByText('similar1.jpg')).toBeInTheDocument();
    expect(screen.getByText('similar3.jpg')).toBeInTheDocument();
  });

  it('shows an empty message when there are no groups', async () => {
    (api.fetchSimilarityGroups as jest.Mock).mockResolvedValue({ groups: [], total: 0 });
    render(<SimilarPhotosGrid jobId="job" threshold={0.99} />);
    await waitFor(() => expect(screen.getByText(/No similar photos found/i)).toBeInTheDocument());
  });

  it('dims results and shows an "Updating results…" overlay while a new threshold loads', async () => {
    (api.fetchSimilarityGroups as jest.Mock)
      .mockResolvedValueOnce({ groups: mockGroups, total: 2 })   // initial load
      .mockReturnValueOnce(new Promise(() => {}));               // next fetch stays pending

    const { rerender } = render(<SimilarPhotosGrid jobId="job" threshold={0.9} />);
    await waitFor(() => expect(screen.getByText('reference1.jpg')).toBeInTheDocument());

    // Changing the threshold kicks off the (pending) refetch → loading state.
    rerender(<SimilarPhotosGrid jobId="job" threshold={0.7} />);

    // Overlay appears, and the previous results remain visible underneath.
    expect(await screen.findByText(/Updating results/i)).toBeInTheDocument();
    expect(screen.getByText('reference1.jpg')).toBeInTheDocument();
  });
});
