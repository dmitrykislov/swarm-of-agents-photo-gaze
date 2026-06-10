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
  const lastCompletedRef = useRef<{ value: number; stalledCount: number }>({ value: -1, stalledCount: 0 });
  const [browserOpen, setBrowserOpen] = useState(false);
  const [browseData, setBrowseData] = useState<BrowseResult | null>(null);
  const [browseLoading, setBrowseLoading] = useState(false);
  const [trashOpen, setTrashOpen] = useState(false);
  const [autoDedupeOpen, setAutoDedupeOpen] = useState(false);

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

  const handleDeleteFolder = async (id: number) => {
    try { await deleteFolder(id); await refreshFolders(); } catch (e) { setFolderError(String(e)); }
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
          // Detect stalled processing: if pending > 0 but completed hasn't
          // changed for 3 consecutive polls (~9s), show the Resume button.
          if (s.pending > 0) {
            if (s.completed === lastCompletedRef.current.value) {
              lastCompletedRef.current.stalledCount++;
            } else {
              lastCompletedRef.current = { value: s.completed, stalledCount: 0 };
            }
            setProcessingStalled(lastCompletedRef.current.stalledCount >= 3);
          } else {
            lastCompletedRef.current = { value: s.completed, stalledCount: 0 };
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
        <h1>Similar Photos Finder</h1>
        <button
          className="app-header__trash"
          onClick={() => setTrashOpen(true)}
          style={{ marginLeft: 'auto', padding: '6px 14px', cursor: 'pointer', fontSize: 13 }}
        >
          Trash
        </button>
      </header>
      <main className="app-main">
        <div className="top-panels">
          <section>
            <h3 style={{ marginTop: 0, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              Photo Folders
              <button onClick={() => openBrowser('/Users')} style={{ padding: '4px 12px', cursor: 'pointer', fontSize: 13 }}>Browse & Add</button>
            </h3>
            {browserOpen && (
              <div style={{ border: '1px solid #ccc', borderRadius: 6, padding: 10, marginBottom: 12, background: '#fafafa', maxHeight: 260, overflow: 'auto' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, fontSize: 13 }}>
                  {browseData?.parent && (
                    <button onClick={() => navigateTo(browseData.parent!)} style={{ padding: '2px 8px', cursor: 'pointer' }}>.. up</button>
                  )}
                  <code style={{ flex: 1, fontSize: 12, color: '#333' }}>{browseData?.path || '/'}</code>
                  {browseData && browseData.image_count > 0 && (
                    <span style={{ fontSize: 12, color: '#888' }}>{browseData.image_count} images</span>
                  )}
                  <button onClick={selectCurrentFolder} style={{ padding: '3px 10px', cursor: 'pointer', fontWeight: 600, fontSize: 12 }}>
                    Select this folder
                  </button>
                  <button onClick={() => { setBrowserOpen(false); setBrowseData(null); }} style={{ padding: '3px 8px', cursor: 'pointer', fontSize: 12 }}>Cancel</button>
                </div>
                {browseLoading ? (
                  <p style={{ color: '#888', margin: 0, fontSize: 13 }}>Loading...</p>
                ) : browseData && browseData.dirs.length === 0 ? (
                  <p style={{ color: '#888', margin: 0, fontSize: 13 }}>No subdirectories.</p>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    {browseData?.dirs.map((d) => (
                      <button
                        key={d.name}
                        onClick={() => navigateTo(browseData.path.replace(/\/$/, '') + '/' + d.name)}
                        style={{ textAlign: 'left', padding: '4px 8px', cursor: 'pointer', background: 'none', border: '1px solid #eee', borderRadius: 4, fontFamily: 'monospace', fontSize: 12 }}
                      >
                        {d.name}/
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
            {folderError && <p style={{ color: '#c00', marginTop: 0, fontSize: 13 }}>{folderError}</p>}
            {folders.length === 0 && !browserOpen ? (
              <p style={{ color: '#888', margin: 0, fontSize: 13 }}>No folders yet. Click "Browse & Add" above.</p>
            ) : (
              <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                {folders.map((f) => (
                  <li key={f.id} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '5px 0', borderTop: '1px solid #eee', fontSize: 13 }}>
                    <span style={{ color: f.is_accessible ? '#090' : '#c00' }}>{f.is_accessible ? '●' : '✗'}</span>
                    <code style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.path}</code>
                    <button
                      onClick={() => handleScanFolder(f.id)}
                      disabled={!f.is_accessible}
                      title="Discover new & changed photos in this folder and start generating embeddings"
                      style={{ padding: '2px 8px', cursor: f.is_accessible ? 'pointer' : 'not-allowed', fontSize: 12 }}
                    >Scan</button>
                    <button onClick={() => handleDeleteFolder(f.id)} title="Remove this folder and all its photos, embeddings, and vectors" style={{ padding: '2px 8px', cursor: 'pointer', fontSize: 12 }}>Remove</button>
                  </li>
                ))}
              </ul>
            )}
            {rescanStatus && <p style={{ color: '#666', fontSize: 12, marginBottom: 0 }}>{rescanStatus}</p>}
            <p style={{ color: '#aaa', fontSize: 11, margin: '8px 0 0', lineHeight: 1.4 }}>
              <strong>Scan</strong> discovers new &amp; changed photos and starts generating embeddings.
              <strong>Remove</strong> deletes the folder and all its data (originals on disk are never touched).
            </p>
          </section>

          <section>
            <h3 style={{ marginTop: 0 }}>Processing Status</h3>
            {stats ? (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8 }}>
                  <Stat label="Photos" value={stats.photos} />
                  <Stat label="Embeddings" value={stats.embeddings} />
                  <Stat label="Completed" value={stats.completed} />
                  <Stat label="Pending" value={stats.pending} />
                  <Stat label="Failed" value={stats.failed} />
                </div>

                {/* Processing status — three states: active / paused / done */}
                {stats.photos > 0 && (
                  <div style={{ marginTop: 12 }}>
                    {stats.pending === 0 ? (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ color: '#090', fontSize: 16 }}>●</span>
                        <span style={{ fontSize: 13, color: '#555' }}>All photos processed. Ready to find duplicates.</span>
                      </div>
                    ) : !processingStalled ? (
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                          <span className="pulse-dot" />
                          <span style={{ fontSize: 13, fontWeight: 600 }}>
                            Processing: {stats.completed}/{stats.photos} ({Math.round(stats.completed / stats.photos * 100)}%)
                          </span>
                          <span style={{ fontSize: 12, color: '#888' }}>
                            {stats.pending} remaining
                          </span>
                        </div>
                        <div style={{ background: '#e0e0e0', borderRadius: 4, height: 8, overflow: 'hidden' }}>
                          <div style={{
                            background: '#4a90e2',
                            height: '100%',
                            borderRadius: 4,
                            width: `${(stats.completed / stats.photos * 100)}%`,
                            transition: 'width 0.5s ease',
                          }} />
                        </div>
                        <div style={{ marginTop: 8 }}>
                          <button
                            onClick={async () => {
                              setRescanStatus('Stopping...');
                              try {
                                const r = await stopProcessing();
                                setRescanStatus(r.message);
                                setProcessingStalled(true);
                                lastCompletedRef.current = { value: -1, stalledCount: 3 };
                              } catch (e) {
                                setRescanStatus(`Failed to stop: ${e instanceof Error ? e.message : e}`);
                              }
                            }}
                            style={{ padding: '6px 14px', cursor: 'pointer', fontSize: 13, fontWeight: 600, color: '#c00', borderColor: '#c00' }}
                          >
                            Stop processing
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                          <span style={{ color: '#e65100', fontSize: 16 }}>●</span>
                          <span style={{ fontSize: 13, fontWeight: 600, color: '#e65100' }}>
                            Paused: {stats.completed}/{stats.photos} ({Math.round(stats.completed / stats.photos * 100)}%)
                          </span>
                          <span style={{ fontSize: 12, color: '#888' }}>
                            {stats.pending} not yet processed
                          </span>
                        </div>
                        <div style={{ background: '#e0e0e0', borderRadius: 4, height: 8, overflow: 'hidden' }}>
                          <div style={{
                            background: '#e65100',
                            height: '100%',
                            borderRadius: 4,
                            width: `${(stats.completed / stats.photos * 100)}%`,
                          }} />
                        </div>
                        <div style={{ marginTop: 8 }}>
                          <button
                            onClick={async () => {
                              setRescanStatus('Queuing...');
                              setProcessingStalled(false);
                              lastCompletedRef.current = { value: -1, stalledCount: 0 };
                              try {
                                const r = await processPending();
                                setRescanStatus(`${r.message}: ${r.queued ?? 0} queued`);
                                if (r.job_id) setJobId(r.job_id);
                              } catch (e) {
                                setRescanStatus(`Failed: ${e instanceof Error ? e.message : e}`);
                              }
                            }}
                            style={{ padding: '6px 14px', cursor: 'pointer', fontSize: 13, fontWeight: 600 }}
                          >
                            Resume processing
                          </button>
                          <span style={{ marginLeft: 8, fontSize: 12, color: '#888' }}>
                            Processing stopped (container restart or all tasks finished). Click to resume.
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </>
            ) : (
              <p style={{ color: '#888', margin: 0 }}>Waiting for backend...</p>
            )}
          </section>
        </div>

        <div className="threshold-row">
          <label htmlFor="threshold-slider">Similarity Threshold:</label>
          <input
            id="threshold-slider"
            type="range" min="0" max="1" step="0.01"
            value={threshold}
            onChange={(e) => handleThresholdChange(parseFloat(e.target.value))}
            style={{ flex: 1 }}
          />
          <span style={{ fontFamily: 'monospace', minWidth: 40 }}>{threshold.toFixed(2)}</span>
          {threshold >= 1.0 && (
            <button
              onClick={() => setAutoDedupeOpen(true)}
              style={{
                marginLeft: 8, padding: '6px 14px', background: '#dc2626',
                color: 'white', border: 'none', borderRadius: 4,
                cursor: 'pointer', fontWeight: 600,
              }}
              title="Sweep all pure-duplicate clusters in one action"
            >
              Auto-deduplicate
            </button>
          )}
        </div>

        <SimilarPhotosGrid jobId={jobId} threshold={threshold} />

        {autoDedupeOpen && (
          <AutoDeduplicateModal
            threshold={threshold}
            onClose={() => setAutoDedupeOpen(false)}
            onCompleted={() => { /* the index already refreshed server-side */ }}
          />
        )}
      </main>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div style={{ textAlign: 'center', padding: 8, background: '#f5f5f5', borderRadius: 4 }}>
      <div style={{ fontSize: 22, fontWeight: 600 }}>{value}</div>
      <div style={{ fontSize: 12, color: '#666' }}>{label}</div>
    </div>
  );
}

export default App;
