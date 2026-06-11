/**
 * Structured error from the API — always has `error`, optionally `detail` and `path`.
 */
export interface ApiError {
  error: string;
  detail?: string;
  path?: string;
}

/**
 * Parse a failed fetch response into a user-friendly error message.
 * Tries to extract the structured JSON body; falls back to status text.
 */
async function parseErrorResponse(response: Response): Promise<string> {
  try {
    const body: ApiError = await response.json();
    if (body.detail) {
      return `${body.error}: ${body.detail}`;
    }
    return body.error || response.statusText;
  } catch {
    return `Server error (${response.status}): ${response.statusText}`;
  }
}

/**
 * Wrap a fetch call with consistent error handling.
 * Network failures and non-2xx responses both throw with a readable message.
 */
async function apiFetch(url: string, options?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(url, options);
  } catch (networkErr) {
    throw new Error(
      `Network error: Unable to reach the server. Please check your connection and try again.`
    );
  }
  if (!response.ok) {
    const message = await parseErrorResponse(response);
    throw new Error(message);
  }
  return response;
}

/**
 * API client for communicating with FastAPI backend.
 * Requests are proxied through package.json proxy configuration.
 */

// Backend base URL, baked at build time (see vite.config.ts):
//   - explicit URL (Docker build arg) → used as-is
//   - "" (native same-origin build)   → relative URLs, so the bundled UI talks
//     to whatever origin/port FastAPI serves it from
//   - undefined (dev / jest)          → localhost:8000 fallback
const _apiBase = process.env.REACT_APP_API_URL;
export const API_BASE_URL = _apiBase == null ? 'http://localhost:8000' : _apiBase;

// WebSocket base: explicit value wins; otherwise (same-origin build) derive it
// from the page's origin at runtime so it follows the actual port.
const _wsBase = process.env.REACT_APP_WS_URL;
const WS_BASE_URL = _wsBase
  ? _wsBase
  : (typeof window !== 'undefined'
      ? `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}`
      : 'ws://localhost:8000');

export interface HealthResponse {
  status: string;
}

export interface ProgressUpdate {
  job_id: string;
  status: string;
  percentage: number;
  processed_photos: number;
  total_photos: number;
  eta_seconds: number | null;
}

export interface UserPreferences {
  id: number;
  username: string;
  email: string;
  preferred_embedding_model: string;
  enable_auto_processing: boolean;
  threshold_setting: number;
}

export interface ThresholdResponse {
  threshold_setting: number;
}

export interface FolderValidationResponse {
  valid: boolean;
  error?: string;
  path: string;
  photo_count?: number;
}

/**
 * Fetch health status from FastAPI backend.
 * @returns Promise resolving to health status object
 * @throws Error if request fails
 */
export interface ProcessingStats {
  photos: number;
  embeddings: number;
  completed: number;
  pending: number;
  failed: number;
  similarity_index?: {
    last_recompute_at: string | null;
    vectors_in_index: number;
    edges_in_index: number;
    recompute_running: boolean;
  };
}

export async function fetchStats(): Promise<ProcessingStats> {
  const response = await fetch(`${API_BASE_URL}/stats`);
  if (!response.ok) {
    throw new Error(`Failed to fetch stats: ${response.statusText}`);
  }
  return response.json();
}

export interface FolderEntry {
  id: number;
  path: string;
  is_accessible: boolean;
  supported_formats_found: string[];
  created_at?: string;
}

export interface BrowseResult {
  path: string;
  parent: string | null;
  dirs: { name: string; type: string }[];
  image_count: number;
}

export async function browsePath(path: string): Promise<BrowseResult> {
  const response = await fetch(`${API_BASE_URL}/browse?path=${encodeURIComponent(path)}`);
  if (!response.ok) throw new Error(`Failed to browse: ${response.statusText}`);
  return response.json();
}

export async function listFolders(): Promise<FolderEntry[]> {
  const response = await fetch(`${API_BASE_URL}/folders`);
  if (!response.ok) throw new Error(`Failed to list folders: ${response.statusText}`);
  return response.json();
}

export async function addFolder(path: string): Promise<FolderEntry> {
  const response = await fetch(`${API_BASE_URL}/folders`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  });
  if (!response.ok) throw new Error(`Failed to add folder: ${response.statusText}`);
  return response.json();
}

export interface DeleteFolderResult {
  deleted: number;
  photos_removed: number;
  embeddings_removed: number;
}

export async function deleteFolder(id: number): Promise<DeleteFolderResult> {
  const response = await fetch(`${API_BASE_URL}/folders/${id}`, { method: 'DELETE' });
  if (!response.ok) throw new Error(`Failed to delete folder: ${response.statusText}`);
  return response.json();
}

export async function scanFolder(id: number): Promise<{ job_id?: string; message: string }> {
  const response = await fetch(`${API_BASE_URL}/folders/${id}/scan`, { method: 'POST' });
  if (!response.ok) throw new Error(`Failed to start scan: ${response.statusText}`);
  return response.json();
}

export async function stopProcessing(): Promise<{ message: string; cancelled_jobs: number }> {
  const response = await fetch(`${API_BASE_URL}/stop-processing`, { method: 'POST' });
  if (!response.ok) {
    throw new Error(`Failed to stop processing: ${response.statusText}`);
  }
  return response.json();
}

export async function processPending(): Promise<{ job_id?: string; message: string; queued?: number }> {
  const response = await fetch(`${API_BASE_URL}/process-pending`, { method: 'POST' });
  if (!response.ok) {
    throw new Error(`Failed to start processing: ${response.statusText}`);
  }
  return response.json();
}

export async function triggerRescan(folderPath?: string): Promise<{ job_id?: string; message: string; changes_found?: number }> {
  const url = folderPath
    ? `${API_BASE_URL}/rescan?folder_path=${encodeURIComponent(folderPath)}`
    : `${API_BASE_URL}/rescan`;
  const response = await fetch(url, { method: 'POST' });
  if (!response.ok) {
    throw new Error(`Failed to start rescan: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Connect to WebSocket endpoint for real-time progress updates.
 * @param jobId - Job identifier to track
 * @param onMessage - Callback function for progress updates
 * @param onError - Callback function for errors
 * @returns Function to close the WebSocket connection
 */
export function connectProgressWebSocket(
  jobId: string,
  onMessage: (data: ProgressUpdate) => void,
  onError: (error: string) => void
): () => void {
  const ws = new WebSocket(`${WS_BASE_URL}/ws/progress/${jobId}`);
  
  ws.onopen = () => {
    console.log(`Connected to progress updates for job ${jobId}`);
  };
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data) as ProgressUpdate;
      onMessage(data);
    } catch (err) {
      onError(`Failed to parse progress update: ${err}`);
    }
  };
  
  ws.onerror = (event) => {
    onError(`WebSocket error: ${event}`);
  };
  
  ws.onclose = () => {
    console.log(`Disconnected from progress updates for job ${jobId}`);
  };
  
  // Return cleanup function
  return () => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.close();
    }
  };
}

/**
 * Fetch user preferences from backend.
 * @param username - Username to fetch preferences for
 * @returns Promise resolving to user preferences object
 * @throws Error if request fails
 */
export async function fetchPreferences(username: string): Promise<UserPreferences> {
  // Backend has no /preferences endpoint yet — serve from localStorage.
  const stored = localStorage.getItem(`preferences:${username}`);
  if (stored) {
    return JSON.parse(stored) as UserPreferences;
  }
  throw new Error('No stored preferences');
}

/**
 * Save user preferences to backend.
 * @param preferences - User preferences object to save
 * @returns Promise resolving to saved preferences
 * @throws Error if request fails
 */
export async function savePreferences(preferences: Omit<UserPreferences, 'id'>): Promise<UserPreferences> {
  // Backend has no /preferences endpoint yet — persist to localStorage.
  const saved: UserPreferences = { id: 1, ...preferences };
  localStorage.setItem(`preferences:${preferences.username}`, JSON.stringify(saved));
  return saved;
}

/**
 * Fetch threshold setting for user.
 * @param username - Username to fetch threshold for
 * @returns Promise resolving to threshold response
 * @throws Error if request fails
 */
export async function fetchThreshold(username: string): Promise<ThresholdResponse> {
  const stored = localStorage.getItem(`threshold:${username}`);
  return { threshold_setting: stored ? parseFloat(stored) : 0.5 };
}

/**
 * Save threshold setting for user.
 * @param username - Username to save threshold for
 * @param threshold - Threshold value (0-1)
 * @returns Promise resolving to threshold response
 * @throws Error if request fails
 */
export async function saveThreshold(username: string, threshold: number): Promise<ThresholdResponse> {
  localStorage.setItem(`threshold:${username}`, threshold.toString());
  return { threshold_setting: threshold };
}

/**
 * Validate folder path and check if it contains photos.
 * @param folderPath - Folder path to validate
 * @returns Promise resolving to validation response with photo count
 * @throws Error if request fails
 */
export async function validateFolderPath(folderPath: string): Promise<FolderValidationResponse> {
  const response = await fetch(`${API_BASE_URL}/validate-folder`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ folder_path: folderPath }),
  });
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || `Failed to validate folder: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Search similar photos by job ID and similarity threshold.
 * Filters results based on threshold value for real-time search updates.
 * @param jobId - Job identifier to search
 * @param threshold - Similarity threshold (0-1) to filter results
 * @returns Promise resolving to array of similar photo groups
 * @throws Error if request fails
 */
export interface TrashItem {
  trash_path: string;
  original_path: string | null;
  filename: string;
  trashed_at: string;
  file_size: number | null;
}

export async function listTrash(): Promise<{ items: TrashItem[]; trash_dir: string }> {
  const response = await fetch(`${API_BASE_URL}/trash`);
  if (!response.ok) {
    throw new Error(`Failed to list trash: ${response.statusText}`);
  }
  return response.json();
}

export interface PhotoImageInfo {
  photo_id: number;
  width: number | null;
  height: number | null;
  created_date: string | null;
}

export async function fetchPhotoImageInfo(photoId: number): Promise<PhotoImageInfo> {
  const response = await fetch(`${API_BASE_URL}/photos/${photoId}/image-info`);
  if (!response.ok) {
    throw new Error(`Failed to fetch image info: ${response.statusText}`);
  }
  return response.json();
}

export interface AutoDedupeResponse {
  dry_run: boolean;
  threshold: number;
  folder_path: string;
  groups_processed: number;
  groups_skipped: number;
  to_delete?: number[];
  kept: number[];
  groups?: {
    /** All kept photos in this cluster (every member already inside the
     * keep folder). kept_ids[0] is the "primary" keeper — first by
     * _best_key — and is intended for display emphasis only. */
    kept_ids: number[];
    kept_paths: (string | null)[];
    /** Members outside the keep folder; these are the only ones that
     * actually get trashed. */
    delete_ids: number[];
    delete_paths: (string | null)[];
  }[];
  deleted?: number;
  moved_to_trash?: number;
  errors?: { photo_id?: number; error: string }[] | null;
}

export async function autoDeduplicate(
  folderPath: string,
  threshold: number,
  dryRun: boolean,
): Promise<AutoDedupeResponse> {
  const response = await fetch(`${API_BASE_URL}/auto-deduplicate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ folder_path: folderPath, threshold, dry_run: dryRun }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `Failed: ${response.statusText}`);
  }
  return response.json();
}

export async function recoverFromTrash(trashPaths: string[]): Promise<{
  recovered: number;
  items: { trash_path: string; restored_to: string }[];
  errors?: { trash_path?: string; error: string }[] | null;
}> {
  const response = await fetch(`${API_BASE_URL}/trash/recover`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trash_paths: trashPaths }),
  });
  if (!response.ok) {
    throw new Error(`Failed to recover: ${response.statusText}`);
  }
  return response.json();
}

export async function deduplicatePhotos(photoIds: number[]): Promise<{
  deleted: number;
  moved_to_trash: number;
  trash_dir: string;
  errors?: { photo_id: number; error: string }[] | null;
}> {
  const response = await fetch(`${API_BASE_URL}/deduplicate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ photo_ids: photoIds }),
  });
  if (!response.ok) {
    throw new Error(`Failed to deduplicate: ${response.statusText}`);
  }
  return response.json();
}

export async function searchSimilarPhotos(jobId: string, threshold: number): Promise<any[]> {
  const response = await fetch(`${API_BASE_URL}/similarity-groups?min_similarity=${threshold}`);
  if (!response.ok) {
    throw new Error(`Failed to search similar photos: ${response.statusText}`);
  }
  const data = await response.json();
  return data.groups || [];
}

export interface SimilarityGroupsPage {
  groups: any[];
  total: number;
  skip: number;
  limit: number;
}

/**
 * Fetch ONE page of similarity groups. `total` is the full count at this
 * threshold so the UI can paginate all the way to the last page (no cap).
 */
export async function fetchSimilarityGroups(
  threshold: number,
  skip: number,
  limit: number,
): Promise<SimilarityGroupsPage> {
  const response = await fetch(
    `${API_BASE_URL}/similarity-groups?min_similarity=${threshold}&skip=${skip}&limit=${limit}`,
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch similar photos: ${response.statusText}`);
  }
  const data = await response.json();
  return {
    groups: data.groups || [],
    total: data.total ?? (data.groups?.length || 0),
    skip: data.skip ?? skip,
    limit: data.limit ?? limit,
  };
}

/**
 * Fetch all similar-photo groups (no threshold filtering).
 */
export async function fetchSimilarPhotos(_jobId: string): Promise<any[]> {
  const response = await fetch(`${API_BASE_URL}/similarity-groups`);
  if (!response.ok) {
    throw new Error(`Failed to fetch similar photos: ${response.statusText}`);
  }
  const data = await response.json();
  return data.groups || [];
}

export interface ThresholdExampleMatch {
  filename: string;
  similarity_score: number;
}

export interface ThresholdExample {
  id: string;
  threshold: number;
  match_count: number;
  sample_matches: ThresholdExampleMatch[];
}

/**
 * Fetch pre-computed threshold examples for a job to help the user
 * pick a sensible similarity threshold.
 */
export async function fetchThresholdExamples(_jobId: string): Promise<ThresholdExample[]> {
  // Backend does not yet expose a threshold-examples endpoint, so return
  // an empty list rather than throwing and blocking the UI.
  return [];
}
