import React, { useEffect, useState, useRef } from 'react';
import { fetchHealth, connectProgressWebSocket, ProgressUpdate, fetchPreferences, savePreferences, fetchThreshold, saveThreshold, UserPreferences, HealthResponse, fetchStats, triggerRescan, processPending, stopProcessing, ProcessingStats, listFolders, addFolder, deleteFolder, scanFolder, FolderEntry, browsePath, BrowseResult } from './api';
import SimilarPhotosGrid from './components/SimilarPhotosGrid';
import TrashPage from './components/TrashPage';
import AutoDeduplicateModal from './components/AutoDeduplicateModal';
import './App.css';

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressUpdate | null>(null);
  // Empty until a real scan/process job starts. The progress WebSocket
  // effect short-circuits on a falsy jobId, so this avoids opening a
  // connection to a non-existent job on first load (the old default
  // 'test_job_001' did exactly that and surfaced a phantom job).
  const [jobId, setJobId] = useState<string>('');
  const [wsConnected, setWsConnected] = useState(false);
  const [username, setUsername] = useState<string>(() => localStorage.getItem('username') || 'default_user');
  const [preferences, setPreferences] = useState<UserPreferences | null>(null);
  const [threshold, setThreshold] = useState<number>(() => parseFloat(localStorage.getItem('threshold') || '0.95'));
  const [preferencesLoading, setPreferencesLoading] = useState(false);
  const [selectedFolders, setSelectedFolders] = useState<string[]>(() => {
    const stored = localStorage.getItem('selectedFolders');
    return stored ? JSON.parse(stored) : [];
  });
  const thresholdDebounceTimer = useRef<NodeJS.Timeout | null>(null);
  const [stats, setStats] = useState<ProcessingStats | null>(null);
  const [rescanStatus, setRescanStatus] = useState<string>('');
  const [folders, setFolders] = useState<FolderEntry[]>([]);
  const [folderError, setFolderError] = useState<string>('');
  const [processingStalled, setProcessingStalled] = useState(false);
  // Tracks the last time processing made progress (completed/pending moved).
  // "Paused" is only shown after a long no-progress window — not a quick poll
  // count — so slow embedding doesn't falsely flip the UI to Paused.
  const lastProgressRef = useRef<{ completed: number; pending: number; at: number }>({ completed: -1, pending: -1, at: 0 });
  // Last similarity-index rebuild time we've seen, to refresh the grid live.
  const lastIndexAtRef = useRef<string | null>(null);
  const [browserOpen, setBrowserOpen] = useState(false);
  const [browseData, setBrowseData] = useState<BrowseResult | null>(null);
  const [browseLoading, setBrowseLoading] = useState(false);
  const [trashOpen, setTrashOpen] = useState(false);
  const [autoDedupeOpen, setAutoDedupeOpen] = useState(false);
  // Folder pending a delete confirmation (null = no dialog open).
  const [folderToDelete, setFolderToDelete] = useState<FolderEntry | null>(null);
  // Bumped after a folder is removed so the results grid refetches.
  const [gridRefresh, setGridRefresh] = useState(0);

  const refreshFolders = async () => {
    try { setFolders(await listFolders()); } catch (e) { /* backend may be starting */ }
  };

  useEffect(() => {
    refreshFolders();
  }, []);

  const openBrowser = async (startPath?: string) => {
    setBrowserOpen(true);
    setBrowseLoading(true);
    try {
      const data = await browsePath(startPath || '/');
      setBrowseData(data);
    } catch (e) {
      setFolderError(e instanceof Error ? e.message : String(e));
    } finally {
      setBrowseLoading(false);
    }
  };

  const navigateTo = async (path: string) => {
    setBrowseLoading(true);
    try {
      setBrowseData(await browsePath(path));
    } catch (e) {
      setFolderError(e instanceof Error ? e.message : String(e));
    } finally {
      setBrowseLoading(false);
    }
  };

  const selectCurrentFolder = async () => {
    if (!browseData) return;
    setFolderError('');
    try {
      await addFolder(browseData.path);
      await refreshFolders();
      setBrowserOpen(false);
      setBrowseData(null);
    } catch (e) {
      setFolderError(e instanceof Error ? e.message : String(e));
    }
  };

  // Actually performs the deletion — only called after the user confirms in
  // the dialog. The backend cascade-removes every photo under the folder (and
  // its subfolders) plus their embeddings and Qdrant vectors.
  const confirmDeleteFolder = async () => {
    if (!folderToDelete) return;
    const f = folderToDelete;
    setFolderError('');
    setRescanStatus('Removing folder…');
    try {
      const res = await deleteFolder(f.id);
      await refreshFolders();
      setGridRefresh((x) => x + 1);
      setRescanStatus(`Removed “${f.path}” — ${res.photos_removed} photo(s) and ${res.embeddings_removed} embedding(s) deleted.`);
    } catch (e) {
      setFolderError(e instanceof Error ? e.message : String(e));
    } finally {
      setFolderToDelete(null);
    }
  };

  const handleScanFolder = async (id: number) => {
    setRescanStatus('Starting scan...');
    try {
      const res = await scanFolder(id);
      setRescanStatus(res.message + (res.job_id ? ` (job ${res.job_id.slice(0, 8)})` : ''));
      if (res.job_id) setJobId(res.job_id);
    } catch (e) {
      setRescanStatus(`Scan failed: ${e instanceof Error ? e.message : e}`);
    }
  };

  // Poll /stats every 3 seconds so progress is visible live
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const s = await fetchStats();
        if (!cancelled) {
          setStats(s);

          // Show new duplicate groups AS photos are processed: the backend
          // rebuilds the index periodically during a long scan; whenever that
          // timestamp advances, refresh the results grid.
          const idxAt = s.similarity_index?.last_recompute_at ?? null;
          if (idxAt && idxAt !== lastIndexAtRef.current) {
            lastIndexAtRef.current = idxAt;
            setGridRefresh((x) => x + 1);
          }

          // "Paused" only after ~30s of NO progress (neither completed nor
          // pending moved). Slow embedding still counts as progress, so this
          // won't falsely flip to Paused.
          if (s.pending > 0) {
            const r = lastProgressRef.current;
            if (s.completed !== r.completed || s.pending !== r.pending) {
              lastProgressRef.current = { completed: s.completed, pending: s.pending, at: Date.now() };
              setProcessingStalled(false);
            } else if (Date.now() - r.at > 30000) {
              setProcessingStalled(true);
            }
          } else {
            lastProgressRef.current = { completed: s.completed, pending: 0, at: Date.now() };
            setProcessingStalled(false);
          }
        }
      } catch {
        /* ignore transient errors */
      }
    };
    poll();
    const id = setInterval(poll, 3000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  const handleRescan = async () => {
    setRescanStatus('Starting rescan...');
    try {
      const res = await triggerRescan();
      setRescanStatus(res.message + (res.job_id ? ` (job ${res.job_id.slice(0, 8)})` : ''));
      if (res.job_id) setJobId(res.job_id);
    } catch (e) {
      setRescanStatus(`Rescan failed: ${e instanceof Error ? e.message : e}`);
    }
  };

  // Load preferences and threshold on mount and when username changes
  useEffect(() => {
    const loadSessionData = async () => {
      setPreferencesLoading(true);
      try {
        const prefs = await fetchPreferences(username);
        setPreferences(prefs);
        setThreshold(prefs.threshold_setting);
        localStorage.setItem('threshold', prefs.threshold_setting.toString());
      } catch (err) {
        console.error('Failed to load preferences:', err);
        // Create default preferences if not found
        const defaultPrefs: Omit<UserPreferences, 'id'> = {
          username,
          email: `${username}@example.com`,
          preferred_embedding_model: 'clip-vit-base-patch32',
          enable_auto_processing: true,
          threshold_setting: 0.95,
        };
        try {
          const saved = await savePreferences(defaultPrefs);
          setPreferences(saved);
          setThreshold(saved.threshold_setting);
          localStorage.setItem('threshold', saved.threshold_setting.toString());
        } catch (saveErr) {
          console.error('Failed to create default preferences:', saveErr);
        }
      } finally {
        setPreferencesLoading(false);
      }
    };

    loadSessionData();
  }, [username]);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const data = await fetchHealth();
        setHealth(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
        setHealth(null);
      } finally {
        setLoading(false);
      }
    };

    checkHealth();
  }, []);

  // Connect to WebSocket for progress updates when jobId changes
  useEffect(() => {
    if (!jobId) return;
    
    const disconnect = connectProgressWebSocket(
      jobId,
      (data) => {
        setProgress(data);
        setWsConnected(true);
        if (data.status === 'not_found') {
          setWsConnected(false);
        }
      },
      (err) => {
        console.error('Progress update error:', err);
        setWsConnected(false);
      }
    );
    
    return () => {
      disconnect();
      setWsConnected(false);
    };
  }, [jobId]);

  const formatETA = (seconds: number | null): string => {
    if (seconds === null || seconds === undefined) return 'Calculating...';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const minutes = Math.round(seconds / 60);
    return `${minutes}m`;
  };

  const handleThresholdChange = async (newThreshold: number) => {
    setThreshold(newThreshold);
    localStorage.setItem('threshold', newThreshold.toString());
    
    // Debounce threshold save to DB (500ms) to avoid excessive API calls
    if (thresholdDebounceTimer.current) {
      clearTimeout(thresholdDebounceTimer.current);
    }
    thresholdDebounceTimer.current = setTimeout(async () => {
      try {
        await saveThreshold(username, newThreshold);
      } catch (err) {
        console.error('Failed to save threshold:', err);
      }
    }, 500);
  };

  const handleUsernameChange = (newUsername: string) => {
    setUsername(newUsername);
    localStorage.setItem('username', newUsername);
  };

  const handleFolderSelect = (folders: string[]) => {
    setSelectedFolders(folders);
    localStorage.setItem('selectedFolders', JSON.stringify(folders));
  };

  if (trashOpen) {
    return (
      <div className="app">
        <TrashPage onClose={() => setTrashOpen(false)} />
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <span className="app-header__brand">
          <span className="app-header__logo" aria-hidden="true">◈</span>
          Photo Gaze
        </span>
        <span className="app-header__spacer" />
        <button className="btn btn--ghost" onClick={() => setTrashOpen(true)}>
          🗑 Trash
        </button>
      </header>
      <main className="app-main">
        <div className="top-panels">
          <section className="card">
            <div className="card__head">
              <h2 className="card__title">Photo Folders</h2>
              <button className="btn btn--primary btn--sm" onClick={() => openBrowser('/Users')}>
                + Browse &amp; Add
              </button>
            </div>
            {browserOpen && (
              <div className="browser">
                <div className="browser__bar">
                  {browseData?.parent && (
                    <button className="btn btn--sm" onClick={() => navigateTo(browseData.parent!)}>↑ Up</button>
                  )}
                  <code className="browser__path">{browseData?.path || '/'}</code>
                  {browseData && browseData.image_count > 0 && (
                    <span className="browser__count">{browseData.image_count} images</span>
                  )}
                  <button className="btn btn--primary btn--sm" onClick={selectCurrentFolder}>
                    Select this folder
                  </button>
                  <button className="btn btn--ghost btn--sm" onClick={() => { setBrowserOpen(false); setBrowseData(null); }}>Cancel</button>
                </div>
                {browseLoading ? (
                  <p className="muted">Loading…</p>
                ) : browseData && browseData.dirs.length === 0 ? (
                  <p className="muted">No subdirectories.</p>
                ) : (
                  <div className="browser__list">
                    {browseData?.dirs.map((d) => (
                      <button
                        key={d.name}
                        className="browser__dir"
                        onClick={() => navigateTo(browseData.path.replace(/\/$/, '') + '/' + d.name)}
                      >
                        📁 {d.name}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
            {folderError && <p className="error-text">{folderError}</p>}
            {folders.length === 0 && !browserOpen ? (
              <p className="muted">No folders yet. Click “Browse &amp; Add” to pick one.</p>
            ) : (
              <ul className="folder-list">
                {folders.map((f) => (
                  <li key={f.id} className="folder-item">
                    <span
                      className={`dot ${f.is_accessible ? 'dot--ok' : 'dot--bad'}`}
                      title={f.is_accessible ? 'Accessible' : 'Not accessible'}
                    />
                    <code className="folder-item__path" title={f.path}>{f.path}</code>
                    <button
                      className="btn btn--sm"
                      onClick={() => handleScanFolder(f.id)}
                      disabled={!f.is_accessible}
                      title="Discover new & changed photos in this folder and start generating embeddings"
                    >Scan</button>
                    <button
                      className="btn btn--ghost btn--sm"
                      onClick={() => setFolderToDelete(f)}
                      title="Remove this folder and all its photos, embeddings, and vectors"
                    >Remove</button>
                  </li>
                ))}
              </ul>
            )}
            {rescanStatus && <p className="status-text">{rescanStatus}</p>}
            <p className="hint">
              <strong>Scan</strong> discovers new &amp; changed photos and starts generating embeddings.{' '}
              <strong>Remove</strong> deletes the folder and all its data (originals on disk are never touched).
            </p>
          </section>

          <section className="card">
            <div className="card__head">
              <h2 className="card__title">Processing Status</h2>
              {stats && stats.photos > 0 && stats.pending === 0 && (
                <span className="pill pill--ok">● Ready</span>
              )}
            </div>
            {stats ? (
              <>
                <div className="stat-grid">
                  <Stat label="Photos" value={stats.photos} />
                  <Stat label="Embeddings" value={stats.embeddings} />
                  <Stat label="Completed" value={stats.completed} accent />
                  <Stat label="Pending" value={stats.pending} />
                  <Stat label="Failed" value={stats.failed} warn={stats.failed > 0} />
                </div>

                {/* Processing status — three states: active / paused / done */}
                {stats.photos > 0 && (
                  <div style={{ marginTop: 14 }}>
                    {stats.pending === 0 ? (
                      <div className="status-row">
                        <span className="dot dot--ok" />
                        <span className="status-row__sub">All photos processed — ready to find duplicates.</span>
                      </div>
                    ) : !processingStalled ? (
                      <div>
                        <div className="status-row">
                          <span className="pulse-dot" />
                          <span className="status-row__title">
                            Processing {stats.completed}/{stats.photos} ({Math.round(stats.completed / stats.photos * 100)}%)
                          </span>
                          <span className="status-row__sub">{stats.pending} remaining</span>
                        </div>
                        <div className="progress">
                          <div className="progress__fill" style={{ width: `${(stats.completed / stats.photos * 100)}%` }} />
                        </div>
                        <div style={{ marginTop: 10 }}>
                          <button
                            className="btn btn--danger btn--sm"
                            onClick={async () => {
                              setRescanStatus('Stopping...');
                              try {
                                const r = await stopProcessing();
                                setRescanStatus(r.message);
                                setProcessingStalled(true);
                                // Force the paused state to stick: mark "no progress since long ago".
                                lastProgressRef.current = { completed: stats?.completed ?? -1, pending: stats?.pending ?? 0, at: 0 };
                              } catch (e) {
                                setRescanStatus(`Failed to stop: ${e instanceof Error ? e.message : e}`);
                              }
                            }}
                          >
                            Stop processing
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div className="status-row">
                          <span className="dot" style={{ background: 'var(--warning)' }} />
                          <span className="status-row__title" style={{ color: 'var(--warning)' }}>
                            Paused {stats.completed}/{stats.photos} ({Math.round(stats.completed / stats.photos * 100)}%)
                          </span>
                          <span className="status-row__sub">{stats.pending} not yet processed</span>
                        </div>
                        <div className="progress">
                          <div className="progress__fill progress__fill--warn" style={{ width: `${(stats.completed / stats.photos * 100)}%` }} />
                        </div>
                        <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                          <button
                            className="btn btn--primary btn--sm"
                            onClick={async () => {
                              setRescanStatus('Queuing...');
                              setProcessingStalled(false);
                              lastProgressRef.current = { completed: -1, pending: -1, at: Date.now() };
                              try {
                                const r = await processPending();
                                setRescanStatus(`${r.message}: ${r.queued ?? 0} queued`);
                                if (r.job_id) setJobId(r.job_id);
                              } catch (e) {
                                setRescanStatus(`Failed: ${e instanceof Error ? e.message : e}`);
                              }
                            }}
                          >
                            Resume processing
                          </button>
                          <span className="status-row__sub">Processing stopped (restart or all tasks finished). Click to resume.</span>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </>
            ) : (
              <p className="muted">Waiting for backend…</p>
            )}
          </section>
        </div>

        <div className="threshold-bar">
          <label className="threshold-bar__label" htmlFor="threshold-slider">Similarity threshold</label>
          <input
            id="threshold-slider"
            type="range" min="0" max="1" step="0.01"
            value={threshold}
            onChange={(e) => handleThresholdChange(parseFloat(e.target.value))}
          />
          <span className="threshold-bar__value">{threshold.toFixed(2)}</span>
          {threshold >= 1.0 && (
            <button
              className="btn btn--danger"
              onClick={() => setAutoDedupeOpen(true)}
              title="Sweep all pure-duplicate clusters in one action"
            >
              Auto-deduplicate
            </button>
          )}
        </div>

        <SimilarPhotosGrid jobId={jobId} threshold={threshold} refreshSignal={gridRefresh} />

        {autoDedupeOpen && (
          <AutoDeduplicateModal
            threshold={threshold}
            onClose={() => setAutoDedupeOpen(false)}
            onCompleted={() => setGridRefresh((x) => x + 1)}
          />
        )}

        {folderToDelete && (
          <div className="modal-overlay" role="dialog" aria-modal="true" onClick={() => setFolderToDelete(null)}>
            <div className="modal" onClick={(e) => e.stopPropagation()}>
              <h3 className="modal__title">Remove this folder?</h3>
              <p className="modal__body">
                Every photo under <code>{folderToDelete.path}</code> — including subfolders — will be
                removed from the database along with its embeddings and vectors.
                <br /><br />
                <strong>Your original files on disk are not touched.</strong> This only clears them from Photo&nbsp;Gaze.
              </p>
              <div className="modal__actions">
                <button className="btn" onClick={() => setFolderToDelete(null)}>Cancel</button>
                <button className="btn btn--danger" onClick={confirmDeleteFolder}>Remove folder</button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

function Stat({ label, value, accent, warn }: { label: string; value: number; accent?: boolean; warn?: boolean }) {
  return (
    <div className={`stat${accent ? ' stat--accent' : ''}${warn ? ' stat--warn' : ''}`}>
      <div className="stat__value">{value.toLocaleString()}</div>
      <div className="stat__label">{label}</div>
    </div>
  );
}

export default App;
