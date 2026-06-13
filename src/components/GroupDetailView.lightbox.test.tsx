import React from 'react';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import GroupDetailView from './GroupDetailView';
import * as api from '../api';

jest.mock('../api');

const mockFetchPhotoImageInfo = api.fetchPhotoImageInfo as jest.MockedFunction<
  typeof api.fetchPhotoImageInfo
>;
const mockDedupe = api.deduplicatePhotos as jest.MockedFunction<
  typeof api.deduplicatePhotos
>;

/**
 * Lightbox interaction tests for GroupDetailView.
 *
 * Coverage targets these recently-shipped behaviors:
 *   1. The keep/delete toggle is available for the auto-picked Best photo
 *      from full-screen too — user can override the Best's "always keep"
 *      stance without first promoting another photo.
 *   2. The "★ Mark as Best" button promotes the visible photo to the
 *      cluster's reference and un-marks it from deletion if it had been
 *      selected.
 *   3. Image info (width / height / created_date) is lazy-loaded via
 *      /photos/{id}/image-info when the lightbox opens, NOT eagerly in
 *      the slider's hot path.
 */

const ref = {
  photo_id: 1, filename: 'best.jpg',
  path: 'http://x/thumbnails/1',
  similarity_score: 1.0,
  file_size: 5_000_000,
  file_path: '/photos/best.jpg',
  mime_type: 'image/jpeg',
  uploaded_at: '2024-01-01T00:00:00',
  width: undefined, height: undefined, created_date: undefined,
};
const other = {
  photo_id: 2, filename: 'other.jpg',
  path: 'http://x/thumbnails/2',
  similarity_score: 0.95,
  file_size: 1_000_000,
  file_path: '/photos/other.jpg',
  mime_type: 'image/jpeg',
  uploaded_at: '2024-01-01T00:00:00',
  width: undefined, height: undefined, created_date: undefined,
};

const group = {
  group_id: 'g1',
  reference_photo: ref,
  similar_photos: [other],
  best_reasons: ['Largest file: 5 MB'],
};

function openLightboxOn(filename: string) {
  // The photo card's <img alt={filename} /> opens the lightbox on click.
  // Use the "detail-image" class to disambiguate from the lightbox's
  // own <img className="lightbox-image" /> once it's open.
  const imgs = screen.getAllByAltText(filename);
  const detailImg = imgs.find(img => img.className.includes('detail-image'));
  if (!detailImg) throw new Error(`no detail-image for ${filename}`);
  return userEvent.click(detailImg);
}

function lightboxIsOpen() {
  // The "← → to navigate · Esc to close" hint is unique to the lightbox.
  return screen.queryByText(/← → to navigate/) !== null;
}


describe('GroupDetailView — lightbox', () => {
  beforeEach(() => {
    mockFetchPhotoImageInfo.mockReset();
    mockDedupe.mockReset();
    mockFetchPhotoImageInfo.mockResolvedValue({
      photo_id: 0, width: null, height: null, created_date: null,
    });
  });

  it('opens the lightbox on the clicked photo', async () => {
    render(<GroupDetailView group={group} onClose={() => {}} />);
    expect(lightboxIsOpen()).toBe(false);
    await openLightboxOn('other.jpg');
    expect(lightboxIsOpen()).toBe(true);
  });

  it('shows "★ Mark as Best" only on photos that are not currently Best', async () => {
    render(<GroupDetailView group={group} onClose={() => {}} />);

    // Open the auto-picked Best (photo 1). Scope to the lightbox — the card
    // grid behind it legitimately has Mark-as-Best buttons for other photos.
    await openLightboxOn('best.jpg');
    const lb = screen.getByTestId('lightbox');
    expect(within(lb).queryByText(/Mark as Best/i)).not.toBeInTheDocument();
  });

  it('Mark-as-Best on a non-Best photo flips the badge and unmarks delete', async () => {
    mockFetchPhotoImageInfo.mockResolvedValue({
      photo_id: 2, width: null, height: null, created_date: null,
    });
    render(<GroupDetailView group={group} onClose={() => {}} />);
    await openLightboxOn('other.jpg');
    const lb = screen.getByTestId('lightbox');

    // Initial state: this photo is NOT Best, was auto-selected for
    // deletion, so the badge says "DELETING — click to keep".
    expect(await within(lb).findByText(/DELETING/i)).toBeInTheDocument();

    // Click "Mark as Best" (the lightbox badge has an action-describing
    // aria-label, so match by its visible text).
    await userEvent.click(within(lb).getByText(/Mark as Best/i));

    // After promotion: photo 2 is now Best AND no longer marked for deletion.
    await waitFor(() =>
      expect(within(lb).queryByText(/DELETING/i)).not.toBeInTheDocument()
    );
    expect(within(lb).getByText(/KEEPING \(Best\)/i)).toBeInTheDocument();
  });

  it('Best photo can be toggled to delete from the lightbox', async () => {
    render(<GroupDetailView group={group} onClose={() => {}} />);
    await openLightboxOn('best.jpg');
    const lb = screen.getByTestId('lightbox');

    // The Best photo's keep badge is a clickable button.
    await userEvent.click(await within(lb).findByText(/KEEPING \(Best\)/i));

    // After clicking, the Best is now marked for deletion.
    await waitFor(() =>
      expect(within(lb).getByText(/DELETING.*click to keep/i)).toBeInTheDocument()
    );
  });

  it('lazy-loads image info via /photos/{id}/image-info on lightbox open', async () => {
    mockFetchPhotoImageInfo.mockResolvedValueOnce({
      photo_id: 1, width: 1920, height: 1080,
      created_date: '2024-06-15T10:00:00',
    });
    render(<GroupDetailView group={group} onClose={() => {}} />);
    await openLightboxOn('best.jpg');

    await waitFor(() =>
      expect(mockFetchPhotoImageInfo).toHaveBeenCalledWith(1)
    );
    // Resolution rendered after lazy load.
    expect(await screen.findByText(/1920×1080/)).toBeInTheDocument();
  });

  it('does not refetch image-info when the lightbox reopens the same photo', async () => {
    mockFetchPhotoImageInfo.mockResolvedValue({
      photo_id: 1, width: 800, height: 600, created_date: null,
    });
    render(<GroupDetailView group={group} onClose={() => {}} />);

    // Open + close + open again
    await openLightboxOn('best.jpg');
    await waitFor(() => expect(mockFetchPhotoImageInfo).toHaveBeenCalledTimes(1));
    // Close via Escape
    await userEvent.keyboard('{Escape}');
    // Reopen
    await openLightboxOn('best.jpg');
    // Cache hit — no second fetch
    await waitFor(() => expect(mockFetchPhotoImageInfo).toHaveBeenCalledTimes(1));
  });

  it('handles image-info fetch failure gracefully (no crash, just no resolution)', async () => {
    mockFetchPhotoImageInfo.mockRejectedValueOnce(new Error('5xx from server'));
    render(<GroupDetailView group={group} onClose={() => {}} />);
    await openLightboxOn('best.jpg');

    // Lightbox still rendered the rest of the metadata bar; the
    // navigation hint proves the lightbox is open.
    expect(lightboxIsOpen()).toBe(true);
    // No 1920×1080 anywhere — but the page didn't crash.
  });
});
