import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import SimilarPhotosGrid from './SimilarPhotosGrid';
import * as api from '../api';

jest.mock('../api');

describe('SimilarPhotosGrid Component', () => {
  const mockGroups = [
    {
      group_id: 'group_1',
      reference_photo: {
        photo_id: 1,
        filename: 'reference1.jpg',
        path: '/photos/reference1.jpg',
        quality_score: 0.95,
      },
      similar_photos: [
        {
          photo_id: 2,
          filename: 'similar1.jpg',
          path: '/photos/similar1.jpg',
          similarity_score: 0.92,
          quality_score: 0.88,
        },
        {
          photo_id: 3,
          filename: 'similar2.jpg',
          path: '/photos/similar2.jpg',
          similarity_score: 0.85,
          quality_score: 0.75,
        },
      ],
    },
    {
      group_id: 'group_2',
      reference_photo: {
        photo_id: 4,
        filename: 'reference2.jpg',
        path: '/photos/reference2.jpg',
        quality_score: 0.82,
      },
      similar_photos: [
        {
          photo_id: 5,
          filename: 'similar3.jpg',
          path: '/photos/similar3.jpg',
          similarity_score: 0.78,
          quality_score: 0.70,
        },
      ],
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    (api.fetchSimilarPhotos as jest.Mock).mockResolvedValue(mockGroups);
  });

  test('renders grid container with title', async () => {
    render(<SimilarPhotosGrid jobId="test_job" />);
    await waitFor(() => {
      expect(screen.getByText(/Similar Photos/)).toBeInTheDocument();
    });
  });

  test('displays correct number of groups', async () => {
    render(<SimilarPhotosGrid jobId="test_job" />);
    await waitFor(() => {
      expect(screen.getByText('Similar Photos (2 groups)')).toBeInTheDocument();
    });
  });

  test('displays group headers with match counts', async () => {
    render(<SimilarPhotosGrid jobId="test_job" />);
    await waitFor(() => {
      expect(screen.getByText('2 matches')).toBeInTheDocument();
      expect(screen.getByText('1 matches')).toBeInTheDocument();
    });
  });

  test('displays reference photo filenames', async () => {
    render(<SimilarPhotosGrid jobId="test_job" />);
    await waitFor(() => {
      expect(screen.getByText('reference1.jpg')).toBeInTheDocument();
      expect(screen.getByText('reference2.jpg')).toBeInTheDocument();
    });
  });

  test('displays similar photo filenames', async () => {
    render(<SimilarPhotosGrid jobId="test_job" />);
    await waitFor(() => {
      expect(screen.getByText('similar1.jpg')).toBeInTheDocument();
      expect(screen.getByText('similar2.jpg')).toBeInTheDocument();
      expect(screen.getByText('similar3.jpg')).toBeInTheDocument();
    });
  });

  test('triggers search when threshold prop changes and updates results', async () => {
    const mockSearchSimilarPhotos = jest.fn().mockResolvedValue(mockGroups);
    (api.searchSimilarPhotos as jest.Mock) = mockSearchSimilarPhotos;
    
    const { rerender } = render(
      <SimilarPhotosGrid jobId="test_job" threshold={0.5} />
    );
    
    await waitFor(() => {
      expect(screen.getByText('Similar Photos (2 groups)')).toBeInTheDocument();
    });
    
    // Change threshold prop
    rerender(<SimilarPhotosGrid jobId="test_job" threshold={0.75} />);
    
    // Verify search was called with new threshold
    await waitFor(() => {
      expect(mockSearchSimilarPhotos).toHaveBeenCalledWith('test_job', 0.75);
    });
  });

  test('displays similarity scores for similar photos', async () => {
    render(<SimilarPhotosGrid jobId="test_job" />);
    await waitFor(() => {
      expect(screen.getByText('92.0%')).toBeInTheDocument();
      expect(screen.getByText('85.0%')).toBeInTheDocument();
      expect(screen.getByText('78.0%')).toBeInTheDocument();
    });
  });

  test('displays quality indicators with correct labels', async () => {
    render(<SimilarPhotosGrid jobId="test_job" />);
    await waitFor(() => {
      const qualityLabels = screen.getAllByText('Quality:');
      expect(qualityLabels.length).toBeGreaterThan(0);
    });
  });

  test('displays quality scores as percentages', async () => {
    render(<SimilarPhotosGrid jobId="test_job" />);
    await waitFor(() => {
      expect(screen.getByText('Excellent (95.0%)')).toBeInTheDocument();
      expect(screen.getByText('Good (88.0%)')).toBeInTheDocument();
      expect(screen.getByText('Good (82.0%)')).toBeInTheDocument();
    });
  });

  test('renders thumbnail images with lazy loading', async () => {
    render(<SimilarPhotosGrid jobId="test_job" />);
    await waitFor(() => {
      const images = screen.getAllByRole('img');
      expect(images.length).toBeGreaterThan(0);
      images.forEach((img) => {
        expect(img).toHaveClass('thumbnail');
      });
    });
  });

  test('shows loading state initially', () => {
    (api.fetchSimilarPhotos as jest.Mock).mockImplementation(
      () => new Promise(() => {}) // Never resolves
    );
    render(<SimilarPhotosGrid jobId="test_job" />);
    expect(screen.getByText('Loading similar photos...')).toBeInTheDocument();
  });

  test('displays error message when fetch fails', async () => {
    (api.fetchSimilarPhotos as jest.Mock).mockRejectedValue(
      new Error('Network error')
    );
    render(<SimilarPhotosGrid jobId="test_job" />);
    await waitFor(() => {
      expect(screen.getByText(/Error: Network error/)).toBeInTheDocument();
    });
  });

  test('shows message when no jobId provided', () => {
    render(<SimilarPhotosGrid jobId="" />);
    expect(
      screen.getByText('No job selected. Process a job to view similar photos.')
    ).toBeInTheDocument();
  });

  test('shows message when no similar photos found', async () => {
    (api.fetchSimilarPhotos as jest.Mock).mockResolvedValue([]);
    render(<SimilarPhotosGrid jobId="test_job" />);
    await waitFor(() => {
      expect(screen.getByText('No similar photos found.')).toBeInTheDocument();
    });
  });

  test('opens GroupDetailView overlay when clicking a group', async () => {
    render(<SimilarPhotosGrid jobId="test_job" />);
    await waitFor(() => {
      expect(screen.getByText('Similar Photos (2 groups)')).toBeInTheDocument();
    });
    // Click the first group container
    const groupContainers = screen.getAllByRole('button');
    fireEvent.click(groupContainers[0]);
    // GroupDetailView overlay should appear with 'Group Detail' title
    await waitFor(() => {
      expect(screen.getByText('Group Detail')).toBeInTheDocument();
    });
  });

  test('closes GroupDetailView overlay when close button is clicked', async () => {
    render(<SimilarPhotosGrid jobId="test_job" />);
    await waitFor(() => {
      expect(screen.getByText('Similar Photos (2 groups)')).toBeInTheDocument();
    });
    const groupContainers = screen.getAllByRole('button');
    fireEvent.click(groupContainers[0]);
    await waitFor(() => {
      expect(screen.getByText('Group Detail')).toBeInTheDocument();
    });
    // Click close button
    const closeButton = screen.getByLabelText('Close');
    fireEvent.click(closeButton);
    await waitFor(() => {
      expect(screen.queryByText('Group Detail')).not.toBeInTheDocument();
    });
  });

  test('fetches photos when jobId changes', async () => {
    const { rerender } = render(<SimilarPhotosGrid jobId="job1" />);
    await waitFor(() => {
      expect(api.fetchSimilarPhotos).toHaveBeenCalledWith('job1');
    });
    jest.clearAllMocks();
    (api.fetchSimilarPhotos as jest.Mock).mockResolvedValue(mockGroups);
    rerender(<SimilarPhotosGrid jobId="job2" />);
    await waitFor(() => {
      expect(api.fetchSimilarPhotos).toHaveBeenCalledWith('job2');
    });
  });

  test('searches with threshold when threshold prop changes', async () => {
    jest.mocked(api.searchSimilarPhotos).mockResolvedValue(mockGroups);
    const { rerender } = render(<SimilarPhotosGrid jobId="test_job" threshold={0.5} />);
    await waitFor(() => {
      expect(api.searchSimilarPhotos).toHaveBeenCalledWith('test_job', 0.5);
    });
    jest.clearAllMocks();
    jest.mocked(api.searchSimilarPhotos).mockResolvedValue(mockGroups);
    rerender(<SimilarPhotosGrid jobId="test_job" threshold={0.75} />);
    await waitFor(() => {
      expect(api.searchSimilarPhotos).toHaveBeenCalledWith('test_job', 0.75);
    });
  });

  test('handles image load events for lazy loading', async () => {
    render(<SimilarPhotosGrid jobId="test_job" />);
    await waitFor(() => {
      const images = screen.getAllByRole('img') as HTMLImageElement[];
      expect(images.length).toBeGreaterThan(0);
      images.forEach((img) => {
        expect(img).toHaveClass('loading');
      });
    });
  });

  test('displays reference photo with special styling', async () => {
    render(<SimilarPhotosGrid jobId="test_job" />);
    await waitFor(() => {
      const referenceCards = screen.getAllByRole('img').filter(
        (img) => img.alt === 'reference1.jpg' || img.alt === 'reference2.jpg'
      );
      expect(referenceCards.length).toBe(2);
    });
  });

  test('quality indicator shows correct color for excellent quality', async () => {
    render(<SimilarPhotosGrid jobId="test_job" />);
    await waitFor(() => {
      expect(screen.getByText('Excellent (95.0%)')).toBeInTheDocument();
    });
  });

  test('quality indicator shows correct color for good quality', async () => {
    render(<SimilarPhotosGrid jobId="test_job" />);
    await waitFor(() => {
      expect(screen.getByText('Good (88.0%)')).toBeInTheDocument();
    });
  });

  test('quality indicator shows correct color for fair quality', async () => {
    const fairQualityGroups = [
      {
        group_id: 'group_fair',
        reference_photo: {
          photo_id: 10,
          filename: 'fair.jpg',
          path: '/photos/fair.jpg',
          quality_score: 0.6,
        },
        similar_photos: [],
      },
    ];
    (api.fetchSimilarPhotos as jest.Mock).mockResolvedValue(fairQualityGroups);
    render(<SimilarPhotosGrid jobId="test_job" />);
    await waitFor(() => {
      expect(screen.getByText('Fair (60.0%)')).toBeInTheDocument();
    });
  });

  test('quality indicator shows correct color for poor quality', async () => {
    const poorQualityGroups = [
      {
        group_id: 'group_poor',
        reference_photo: {
          photo_id: 11,
          filename: 'poor.jpg',
          path: '/photos/poor.jpg',
          quality_score: 0.3,
        },
        similar_photos: [],
      },
    ];
    (api.fetchSimilarPhotos as jest.Mock).mockResolvedValue(poorQualityGroups);
    render(<SimilarPhotosGrid jobId="test_job" />);
    await waitFor(() => {
      expect(screen.getByText('Poor (30.0%)')).toBeInTheDocument();
    });
  });
