import React, { useEffect, useState } from 'react';
import { listTrash, recoverFromTrash, revealTrash, TrashItem, API_BASE_URL } from '../api';
import './TrashPage.css';

interface TrashPageProps {
  onClose: () => void;
}

function formatSize(bytes: number | null): string {
  if (bytes == null) return '—';
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(2)} MB`;
  return `${(bytes / 1_000).toFixed(1)} KB`;
}

function formatTrashedAt(ts: string): string {
  // ts is YYYYMMDD_HHMMSS in UTC; render as a readable local-ish stamp
  if (!/^\d{8}_\d{6}$/.test(ts)) return ts;
  const y = ts.slice(0, 4), mo = ts.slice(4, 6), d = ts.slice(6, 8);
  const h = ts.slice(9, 11), mi = ts.slice(11, 13), s = ts.slice(13, 15);
  return `${y}-${mo}-${d} ${h}:${mi}:${s} UTC`;
}

const TrashPage: React.FC<TrashPageProps> = ({ onClose }) => {
  const [items, setItems] = useState<TrashItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [recovering, setRecovering] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [trashDir, setTrashDir] = useState<string>('');

  const openTrashFolder = async () => {
    try {
      await revealTrash();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listTrash();
      setItems(data.items);
      setTrashDir(data.trash_dir);
      // Drop any selections that no longer exist in the new list.
      setSelected(prev => {
        const valid = new Set(data.items.map(i => i.trash_path));
        const next = new Set<string>();
        prev.forEach(p => { if (valid.has(p)) next.add(p); });
        return next;
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { refresh(); }, []);

  const toggleOne = (trashPath: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(trashPath)) next.delete(trashPath);
      else next.add(trashPath);
      return next;
    });
  };

  const toggleAll = () => {
    if (selected.size === items.length) {
      setSelected(new Set());
    } else {
      setSelected(new Set(items.map(i => i.trash_path)));
    }
  };

  const handleRecover = async () => {
    if (selected.size === 0) return;
    setRecovering(true);
    setStatusMessage(null);
    try {
      const result = await recoverFromTrash(Array.from(selected));
      const errCount = result.errors?.length || 0;
      let msg = `Recovered ${result.recovered} photo${result.recovered === 1 ? '' : 's'}`;
      if (errCount > 0) {
        msg += ` · ${errCount} failed (`;
        msg += result.errors!.slice(0, 3).map(e => e.error).join('; ');
        if (errCount > 3) msg += `; …`;
        msg += ')';
      }
      setStatusMessage(msg);
      await refresh();
    } catch (e) {
      setStatusMessage(`Recovery failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setRecovering(false);
    }
  };

  const allSelected = items.length > 0 && selected.size === items.length;

  return (
    <div className="trash-page">
      <header className="trash-page__header">
        <button className="trash-page__back" onClick={onClose} aria-label="Back">←</button>
        <h2>Trash ({items.length})</h2>
        <div className="trash-page__actions">
          <button onClick={refresh} disabled={loading}>Refresh</button>
          <button
            className="trash-page__recover"
            onClick={handleRecover}
            disabled={selected.size === 0 || recovering}
          >
            {recovering ? 'Recovering…' : `Recover ${selected.size} selected`}
          </button>
        </div>
      </header>

      {trashDir && (
        <div className="trash-page__location">
          <span className="muted">Trash folder:</span>{' '}
          <button
            type="button"
            className="trash-page__location-link"
            onClick={openTrashFolder}
            title="Open this folder on your computer"
            style={{
              background: 'none', border: 'none', padding: 0, cursor: 'pointer',
              color: 'var(--accent, #3b82f6)', textDecoration: 'underline',
              font: 'inherit', wordBreak: 'break-all', textAlign: 'left',
            }}
          >
            {trashDir} ↗
          </button>
        </div>
      )}

      {statusMessage && (
        <div className="trash-page__status" role="status">{statusMessage}</div>
      )}

      {error && <div className="trash-page__error">{error}</div>}

      {loading ? (
        <p>Loading trash…</p>
      ) : items.length === 0 ? (
        <p className="trash-page__empty">Trash is empty.</p>
      ) : (
        <>
          <div className="trash-page__select-all">
            <label>
              <input
                type="checkbox"
                checked={allSelected}
                onChange={toggleAll}
                aria-label="Select all"
              />
              {' '}Select all
            </label>
          </div>
          <ul className="trash-page__grid" aria-label="Trashed photos">
            {items.map(item => {
              const isSelected = selected.has(item.trash_path);
              const thumbUrl = `${API_BASE_URL}/trash/thumbnail?path=${encodeURIComponent(item.trash_path)}`;
              return (
                <li key={item.trash_path}>
                  <label className={`trash-card ${isSelected ? 'is-selected' : ''}`}
                         title={item.original_path || ''}>
                    <input
                      type="checkbox"
                      className="trash-card__check"
                      checked={isSelected}
                      onChange={() => toggleOne(item.trash_path)}
                      aria-label={`Select ${item.filename}`}
                    />
                    <div className="trash-card__thumb-wrap">
                      <img
                        src={thumbUrl}
                        alt={item.filename}
                        className="trash-card__thumb"
                        loading="lazy"
                        onError={e => {
                          // Failed to load (e.g. unsupported format) — show
                          // a placeholder so the layout doesn't shift.
                          (e.currentTarget as HTMLImageElement).style.display = 'none';
                        }}
                      />
                    </div>
                    <div className="trash-card__details">
                      <div className="trash-card__filename">{item.filename}</div>
                      <div className="trash-card__path">
                        {item.original_path || <em>unknown</em>}
                      </div>
                      <div className="trash-card__meta">
                        {formatTrashedAt(item.trashed_at)} · {formatSize(item.file_size)}
                      </div>
                    </div>
                  </label>
                </li>
              );
            })}
          </ul>
        </>
      )}
    </div>
  );
};

export default TrashPage;
