import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import axios from 'axios';
import { SimilarityGroupCard } from '../SimilarityGroupCard';

jest.mock('axios');

describe('SimilarityGroupCard', () => {
  const mockGroup = {
    group_id: 'g1',
    similarity_score: 0.95,
    quality_score: 0.8,
    members: [
      {
        photo_id: 1,
        filename: 'photo_1.jpg',
        quality_score: 0.9,
        thumbnail: '/cache/thumb1.jpg',
      },
      {
        photo_id: 2,
        filename: 'photo_2.jpg',
        quality_score: 0.8,
        thumbnail: '/cache/thumb2.jpg',
      },
      {
        photo_id: 3,
        filename: 'photo_3.jpg',
        quality_score: 0.7,
        thumbnail: '/cache/thumb3.jpg',
      },
    ],
  };

  beforeEach(() => {
    jest.clearAllMocks();
    window.confirm = jest.fn(() => true);
  });

  test('renders group with all photos', () => {
    render(<SimilarityGroupCard group={mockGroup} />);
    expect(screen.getByText('Group g1')).toBeInTheDocument();
    expect(screen.getByText('photo_1.jpg')).toBeInTheDocument();
    expect(screen.getByText('photo_2.jpg')).toBeInTheDocument();
    expect(screen.getByText('photo_3.jpg')).toBeInTheDocument();
  });

  test('best image (highest quality_score) is pre-selected', () => {
    render(<SimilarityGroupCard group={mockGroup} />);
    const checkbox1 = screen.getByRole('checkbox', { name: /Best/i });
    const checkbox2 = screen.getByRole('checkbox', { name: /Remove/i });
    
    // Photo 1 has highest quality (0.9), should be disabled (best)
    expect(checkbox1).toBeDisabled();
    // Photo 2 and 3 should be checked by default (not best)
    expect(checkbox2).toBeChecked();
  });

  test('toggling checkbox updates selection', () => {
    render(<SimilarityGroupCard group={mockGroup} />);
    const checkboxes = screen.getAllByRole('checkbox');
    
    // Photo 2 should be checked initially
    expect(checkboxes[1]).toBeChecked();
    
    // Toggle it
    fireEvent.click(checkboxes[1]);
    expect(checkboxes[1]).not.toBeChecked();
  });

  test('deduplicate button sends unchecked photo IDs to backend', async () => {
    axios.post.mockResolvedValue({ status: 200 });
    const onSuccess = jest.fn();
    
    render(<SimilarityGroupCard group={mockGroup} onDeduplicateSuccess={onSuccess} />);
    
    const deduplicateButton = screen.getByRole('button', { name: /Deduplicate/i });
    fireEvent.click(deduplicateButton);
    
    await waitFor(() => {
      expect(axios.post).toHaveBeenCalledWith('/deduplicate', {
        photo_ids: expect.arrayContaining([2, 3]),
      });
    });
  });

  test('confirmation dialog is shown before deduplication', () => {
    axios.post.mockResolvedValue({ status: 200 });
    render(<SimilarityGroupCard group={mockGroup} />);
    
    const deduplicateButton = screen.getByRole('button', { name: /Deduplicate/i });
    fireEvent.click(deduplicateButton);
    
    expect(window.confirm).toHaveBeenCalledWith('Move 2 photo(s) to trash?');
  });

  test('deduplication is cancelled if user rejects confirmation', async () => {
    window.confirm.mockReturnValue(false);
    axios.post.mockResolvedValue({ status: 200 });
    
    render(<SimilarityGroupCard group={mockGroup} />);
    
    const deduplicateButton = screen.getByRole('button', { name: /Deduplicate/i });
    fireEvent.click(deduplicateButton);
    
    await waitFor(() => {
      expect(axios.post).not.toHaveBeenCalled();
    });
  });

  test('error message is displayed on deduplication failure', async () => {
    axios.post.mockRejectedValue({
      response: { data: { error: 'Deduplication failed' } },
    });
    
    render(<SimilarityGroupCard group={mockGroup} />);
    
    const deduplicateButton = screen.getByRole('button', { name: /Deduplicate/i });
    fireEvent.click(deduplicateButton);
    
    await waitFor(() => {
      expect(screen.getByText('Deduplication failed')).toBeInTheDocument();
    });
  });

  test('deduplicate button is disabled when no photos are selected', () => {
    render(<SimilarityGroupCard group={mockGroup} />);
    
    const checkboxes = screen.getAllByRole('checkbox');
    // Uncheck all removable photos
    fireEvent.click(checkboxes[1]);
    fireEvent.click(checkboxes[2]);
    
    const deduplicateButton = screen.getByRole('button', { name: /Deduplicate/i });
    expect(deduplicateButton).toBeDisabled();
  });

  test('onDeduplicateSuccess callback is called after successful deduplication', async () => {
    axios.post.mockResolvedValue({ status: 200 });
    const onSuccess = jest.fn();
    
    render(<SimilarityGroupCard group={mockGroup} onDeduplicateSuccess={onSuccess} />);
    
    const deduplicateButton = screen.getByRole('button', { name: /Deduplicate/i });
    fireEvent.click(deduplicateButton);
    
    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalledWith('g1');
    });
  });
});
