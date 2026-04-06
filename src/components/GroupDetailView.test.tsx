import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import GroupDetailView from './GroupDetailView';
import * as api from '../api';

jest.mock('../api');

describe('GroupDetailView Component', () => {
  const mockGroup = {
    group_id: 'group_1',
    reference_photo: {
      photo_id: 1,
      filename: 'reference.jpg',
      path: '/photos/reference.jpg',
      quality_score: 0.95,
      resolution: '4032x3024',
      file_size: '3.2 MB',
      created_date: '2024-01-15',
    },
    similar_photos: [
      {
        photo_id: 2,
        filename: 'similar1.jpg',
        path: '/photos/similar1.jpg',
        quality_score: 0.88,
        resolution: '3840x2160',
        file_size: '2.8 MB',
        created_date: '2024-01-16',
        similarity_score: 0.92,
      },
      {
        photo_id: 3,
        filename: 'similar2.jpg',
        path: '/photos/similar2.jpg',
        quality_score: 0.75,
        resolution: '2560x1920',
        file_size: '1.5 MB',
        created_date: '2024-01-17',
        similarity_score: 0.85,
      },
    ],
  };

  const mockOnClose = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders detail view with title', () => {
    render(<GroupDetailView group={mockGroup} onClose={mockOnClose} />);
    expect(screen.getByText('Group Detail')).toBeInTheDocument();
  });

  test('displays all photos in the group', () => {
    render(<GroupDetailView group={mockGroup} onClose={mockOnClose} />);
    expect(screen.getByAltText('reference.jpg')).toBeInTheDocument();
    expect(screen.getByAltText('similar1.jpg')).toBeInTheDocument();
    expect(screen.getByAltText('similar2.jpg')).toBeInTheDocument();
  });

  test('displays metadata for each photo', () => {
    render(<GroupDetailView group={mockGroup} onClose={mockOnClose} />);
    expect(screen.getByText('reference.jpg')).toBeInTheDocument();
    expect(screen.getByText(/Resolution: 4032x3024/)).toBeInTheDocument();
    expect(screen.getByText(/Size: 3.2 MB/)).toBeInTheDocument();
    expect(screen.getByText(/Created: 2024-01-15/)).toBeInTheDocument();
    expect(screen.getByText(/Path: \/photos\/reference.jpg/)).toBeInTheDocument();
  });

  test('marks the best photo with indicator', () => {
    render(<GroupDetailView group={mockGroup} onClose={mockOnClose} />);
    const bestIndicator = screen.getByText('★ Best');
    expect(bestIndicator).toBeInTheDocument();
    // Best photo should be the reference (highest quality_score: 0.95)
    const bestCard = bestIndicator.closest('.photo-card');
    expect(bestCard).toHaveClass('best-photo');
  });

  test('displays quality scores as percentages', () => {
    render(<GroupDetailView group={mockGroup} onClose={mockOnClose} />);
    expect(screen.getByText('Quality: 95.0%')).toBeInTheDocument();
    expect(screen.getByText('Quality: 88.0%')).toBeInTheDocument();
    expect(screen.getByText('Quality: 75.0%')).toBeInTheDocument();
  });

  test('allows selecting photos for deletion', () => {
    render(<GroupDetailView group={mockGroup} onClose={mockOnClose} />);
    const checkboxes = screen.getAllByRole('checkbox');
    // First checkbox (best photo) should be disabled
    expect(checkboxes[0]).toBeDisabled();
    // Other checkboxes should be enabled
    expect(checkboxes[1]).not.toBeDisabled();
    fireEvent.click(checkboxes[1]);
    expect(checkboxes[1]).toBeChecked();
  });

  test('disables best photo checkbox', () => {
    render(<GroupDetailView group={mockGroup} onClose={mockOnClose} />);
    const checkboxes = screen.getAllByRole('checkbox');
    // Best photo (reference with 0.95 quality) should be disabled
    expect(checkboxes[0]).toBeDisabled();
    expect(checkboxes[0]).toHaveAttribute('title', 'Cannot delete the best photo');
  });

  test('updates delete button count when photos are selected', () => {
    render(<GroupDetailView group={mockGroup} onClose={mockOnClose} />);
    const checkboxes = screen.getAllByRole('checkbox');
    expect(screen.getByText('Delete Selected (0)')).toBeInTheDocument();
    fireEvent.click(checkboxes[1]);
    expect(screen.getByText('Delete Selected (1)')).toBeInTheDocument();
    fireEvent.click(checkboxes[2]);
    expect(screen.getByText('Delete Selected (2)')).toBeInTheDocument();
  });

  test('calls deduplicatePhotos API when delete button is clicked', async () => {
    (api.deduplicatePhotos as jest.Mock).mockResolvedValue({
      deleted: 2,
      message: 'Photos deleted successfully',
    });
    render(<GroupDetailView group={mockGroup} onClose={mockOnClose} />);
    const checkboxes = screen.getAllByRole('checkbox');
    fireEvent.click(checkboxes[1]);
    fireEvent.click(checkboxes[2]);
    const deleteButton = screen.getByText('Delete Selected (2)');
    fireEvent.click(deleteButton);
    await waitFor(() => {
      expect(api.deduplicatePhotos).toHaveBeenCalledWith([2, 3]);
    });
  });

  test('displays success message after deduplication', async () => {
    (api.deduplicatePhotos as jest.Mock).mockResolvedValue({
      deleted: 1,
      message: 'Photo deleted successfully',
    });
    render(<GroupDetailView group={mockGroup} onClose={mockOnClose} />);
    const checkboxes = screen.getAllByRole('checkbox');
    fireEvent.click(checkboxes[1]);
    const deleteButton = screen.getByText('Delete Selected (1)');
    fireEvent.click(deleteButton);
    await waitFor(() => {
      expect(screen.getByText(/Successfully deleted 1 photo/)).toBeInTheDocument();
    });
  });

  test('displays error message on deduplication failure', async () => {
    (api.deduplicatePhotos as jest.Mock).mockRejectedValue(
      new Error('Network error')
    );
    render(<GroupDetailView group={mockGroup} onClose={mockOnClose} />);
    const checkboxes = screen.getAllByRole('checkbox');
    fireEvent.click(checkboxes[1]);
    const deleteButton = screen.getByText('Delete Selected (1)');
    fireEvent.click(deleteButton);
    await waitFor(() => {
      expect(screen.getByText(/Error: Network error/)).toBeInTheDocument();
    });
  });

  test('closes modal when close button is clicked', () => {
    render(<GroupDetailView group={mockGroup} onClose={mockOnClose} />);
    const closeButton = screen.getByLabelText('Close');
    fireEvent.click(closeButton);
    expect(mockOnClose).toHaveBeenCalled();
  });

  test('closes modal when clicking outside (backdrop)', () => {
    render(<GroupDetailView group={mockGroup} onClose={mockOnClose} />);
    const overlay = screen.getByText('Group Detail').closest('.group-detail-overlay');
    fireEvent.click(overlay!);
    expect(mockOnClose).toHaveBeenCalled();
  });

  test('does not close modal when clicking inside modal content', () => {
    render(<GroupDetailView group={mockGroup} onClose={mockOnClose} />);
    const modal = screen.getByText('Group Detail').closest('.group-detail-modal');
    fireEvent.click(modal!);
    expect(mockOnClose).not.toHaveBeenCalled();
  });

  test('disables delete button when no photos are selected', () => {
    render(<GroupDetailView group={mockGroup} onClose={mockOnClose} />);
    const deleteButton = screen.getByText('Delete Selected (0)') as HTMLButtonElement;
    expect(deleteButton.disabled).toBe(true);
  });

  test('clears selection after successful deduplication', async () => {
    (api.deduplicatePhotos as jest.Mock).mockResolvedValue({
      deleted: 1,
      message: 'Photo deleted',
    });
    render(<GroupDetailView group={mockGroup} onClose={mockOnClose} />);
    const checkboxes = screen.getAllByRole('checkbox');
    fireEvent.click(checkboxes[1]);
    expect(checkboxes[1]).toBeChecked();
    const deleteButton = screen.getByText('Delete Selected (1)');
    fireEvent.click(deleteButton);
    await waitFor(() => {
      expect(screen.getByText('Delete Selected (0)')).toBeInTheDocument();
    });
  });
});
