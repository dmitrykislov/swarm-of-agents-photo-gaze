import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from './App';
import * as api from './api';

jest.mock('./api');

const EMPTY_STATS = {
  photos: 0, completed: 0, pending: 0, failed: 0, embeddings: 0,
};

describe('App Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Mock everything App touches on mount so it can render without a backend.
    (api.fetchHealth as jest.Mock).mockResolvedValue({ status: 'healthy' });
    (api.fetchStats as jest.Mock).mockResolvedValue(EMPTY_STATS);
    (api.listFolders as jest.Mock).mockResolvedValue([]);
    // App reads prefs.threshold_setting → setThreshold; a missing field would
    // make `threshold` undefined and crash threshold.toFixed(2) on render.
    (api.fetchPreferences as jest.Mock).mockResolvedValue({ threshold_setting: 0.9 });
    (api.fetchThreshold as jest.Mock).mockResolvedValue({ threshold_setting: 0.9 });
    (api.connectProgressWebSocket as jest.Mock).mockReturnValue({ close: jest.fn() });
  });

  it('renders the Photo Gaze header', async () => {
    render(<App />);
    expect(await screen.findByText('Photo Gaze')).toBeInTheDocument();
  });

  it('mounts and queries backend health/stats on load', async () => {
    render(<App />);
    await waitFor(() => expect(api.fetchHealth).toHaveBeenCalled());
    expect(api.fetchStats).toHaveBeenCalled();
    expect(api.listFolders).toHaveBeenCalled();
  });

  it('still renders the shell when the backend is unreachable', async () => {
    (api.fetchHealth as jest.Mock).mockRejectedValue(new Error('Network error'));
    (api.fetchStats as jest.Mock).mockRejectedValue(new Error('Network error'));
    render(<App />);
    // The app chrome renders regardless of backend errors.
    expect(await screen.findByText('Photo Gaze')).toBeInTheDocument();
  });
});
