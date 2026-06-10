# Troubleshooting Guide



## Common Issues and Solutions



### Backend Issues



#### 1. "Connection refused" when accessing http://localhost:8000



**Symptoms**: `curl: (7) Failed to connect to localhost port 8000`



**Causes**:

- Backend service is not running

- Port 8000 is already in use

- Firewall blocking connections



**Solutions**:



```bash

# Check if backend is running
docker-compose ps



# If not running, start it

docker-compose up -d fastapi



# Check if port 8000 is in use

lsof -i :8000  # macOS/Linux

netstat -ano | findstr :8000  # Windows



# If port is in use, kill the process or use a different port

kill -9 <PID>

# Or change port in docker-compose.yml

```



---



#### 2. "Health check failed: database connection error"



**Symptoms**: `GET /health` returns 503 with database error



**Causes**:

- PostgreSQL is not running

- DATABASE_URL is incorrect

- Database credentials are wrong

- Network connectivity issue



**Solutions**:



```bash

# Check if PostgreSQL is running
docker-compose ps postgres



# Start PostgreSQL if not running

docker-compose up -d postgres



# Verify DATABASE_URL in .env

cat .env | grep DATABASE_URL



# Test database connection

psql postgresql://postgres:postgres@localhost:5432/app_db



# Check PostgreSQL logs

docker-compose logs postgres

```



---



#### 3. "Qdrant connection error" in logs



**Symptoms**: Backend logs show Qdrant connection failures



**Causes**:

- Qdrant service is not running

- QDRANT_URL is incorrect

- Network connectivity issue



**Solutions**:



```bash

# Check if Qdrant is running
docker-compose ps qdrant



# Start Qdrant if not running

docker-compose up -d qdrant



# Verify QDRANT_URL in .env

cat .env | grep QDRANT_URL



# Test Qdrant connection

curl http://localhost:6333/health



# Check Qdrant logs

docker-compose logs qdrant

```



---



#### 4. "Job stuck in 'processing' state"



**Symptoms**: Job progress doesn't update, WebSocket connection hangs



**Causes**:

- Embedding generation is CPU-bound (no GPU is used; ~40 photos/min in Docker)

- Large number of photos

- Backend process crashed



**Solutions**:



```bash

# Check backend logs for errors
docker-compose logs fastapi | tail -50



# Is it actually progressing? (embeddings are CPU-bound, ~40/min)
curl -s http://localhost:8000/stats | jq '{completed, pending, failed}'



# Restart the backend service
docker-compose restart fastapi



# For very large folders (>10k photos), increase timeout

# Edit docker-compose.yml and increase timeout values

```



---



#### 5. "Out of memory" errors during processing



**Symptoms**: Backend crashes with OOM error, job fails



**Causes**:

- Processing too many photos at once

- Insufficient container memory

- Memory leak in embedding generation



**Solutions**:



```bash

# Increase Docker memory limit in docker-compose.yml

services:

  backend:

    mem_limit: 4g  # Increase from default



# Or process photos in smaller batches

# Split large folders into subfolders



# Monitor memory usage
docker stats fastapi_app

```



---



#### 6. "Migration failed" on startup



**Symptoms**: Backend logs show Alembic migration error



**Causes**:

- Database schema is out of sync

- Migration file is corrupted

- Database permissions issue



**Solutions**:



```bash

# Check migration status
docker-compose exec fastapi alembic current



# View migration history
docker-compose exec fastapi alembic history



# Manually run migrations
docker-compose exec fastapi alembic upgrade head



# If migration is stuck, downgrade and retry
docker-compose exec fastapi alembic downgrade -1
docker-compose exec fastapi alembic upgrade head

```



---



### Frontend Issues



#### 1. "Cannot reach the server" error in UI



**Symptoms**: Frontend shows error message, API calls fail



**Causes**:

- Backend is not running

- REACT_APP_API_URL is incorrect

- CORS policy blocking requests



**Solutions**:



A bare **"Failed to fetch"** in the UI is almost always a port/CORS mismatch.

```bash
# 1) Is the backend up on the configured port? (FASTAPI_PORT, default 8000)
curl http://localhost:8000/health        # -> {"status":"healthy"}

# 2) The UI's backend URL is BAKED INTO THE BUNDLE AT BUILD TIME from
#    REACT_APP_API_URL (a docker build arg derived from FASTAPI_PORT in
#    start.sh) — editing .env at runtime does NOT change it. If you changed
#    FASTAPI_PORT, rebuild the frontend so it points at the new port:
docker compose up -d --build react       # or ./start.sh
```

CORS itself should not be the problem: the backend allows **any
`localhost`/`127.0.0.1` origin on any port** (so the UI works regardless of
`REACT_PORT`). The relevant config in `app/main.py` is:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_extra,  # optional extra origins via CORS_ORIGINS
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

For a non-localhost host, add it via the `CORS_ORIGINS` env var (comma-separated).



---



#### 2. "WebSocket connection failed"



**Symptoms**: Progress bar doesn't update, WebSocket error in console



**Causes**:

- Backend WebSocket handler is not running

- REACT_APP_WS_URL is incorrect

- Firewall blocking WebSocket connections



**Solutions**:



```bash

# Verify REACT_APP_WS_URL in .env
cat .env | grep REACT_APP_WS_URL



# Test WebSocket connection
wscat -c ws://localhost:8000/ws/progress/test-job-id



# Check backend logs for WebSocket errors
docker-compose logs fastapi | grep -i websocket



# Ensure backend is running and healthy
curl http://localhost:8000/health

```



---



#### 3. "Folder path not found" error



**Symptoms**: User enters valid folder path, backend returns 400 error



**Causes**:

- Path doesn't exist on backend filesystem

- Path is relative (not absolute)

- Permissions issue (backend can't read folder)



**Solutions**:



```bash

# Use absolute paths, not relative paths

# ✓ /home/user/Pictures

# ✗ ~/Pictures

# ✗ ./Pictures



# Check if path exists on backend
docker-compose exec fastapi ls -la /path/to/folder



# Check backend permissions
docker-compose exec fastapi stat /path/to/folder



# If using Docker, mount the folder in docker-compose.yml

services:

  backend:

    volumes:

      - /home/user/Pictures:/photos:ro

# Then use /photos in the UI

```



---



#### 4. "No images found" in empty folder



**Symptoms**: User selects folder with images, but backend reports 0 photos



**Causes**:

- Image format not supported (BMP, TIFF, etc.)

- Images are in subdirectories (scanner doesn't recurse)

- File permissions prevent reading images



**Solutions**:



```bash

# Check supported formats in app/folder_scanner.py

# Currently supports: JPEG, PNG, WebP



# Verify images are readable
docker-compose exec fastapi ls -la /path/to/folder/*.jpg



# Check if scanner recurses into subdirectories

# It should find images in nested folders



# Convert unsupported formats
convert image.bmp image.jpg  # ImageMagick

```



---



#### 5. "Threshold slider not working"



**Symptoms**: Slider doesn't respond to input, results don't update



**Causes**:

- JavaScript error in ThresholdInput component

- The /similarity-groups request is failing

- No similarity groups to display (index not built yet, or threshold too high)



**Solutions**:



```bash

# Check browser console for JavaScript errors

# Open DevTools (F12) → Console tab



# Verify the similarity-groups endpoint is working (lower the threshold to
# see more; the index floor is 0.70). Groups are global — no job id needed.
curl -s "http://localhost:8000/similarity-groups?min_similarity=0.85" | jq '.total'



# If 0, check the index is built and photos are embedded
curl -s http://localhost:8000/stats | jq '{embeddings, similarity_index}'

# Run a scan first if there are no embeddings



# Check frontend logs
docker-compose logs react

```



---



### Docker Issues



#### 1. "docker-compose: command not found"



**Solutions**:



```bash

# Install Docker Compose

# macOS: brew install docker-compose

# Linux: sudo apt-get install docker-compose

# Windows: Install Docker Desktop



# Or use docker compose (v2)
docker compose up -d

```



---



#### 2. "Permission denied" when running docker commands



**Solutions**:



```bash

# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker



# Or use sudo
sudo docker-compose up -d

```



---



#### 3. "Disk space full" error



**Solutions**:



```bash

# Check disk usage
df -h



# Clean up Docker images and containers
docker system prune -a



# Remove old backups
rm -rf backups/old-*

```



---



### Performance Issues



#### 1. "Embedding generation is very slow"



**Expected throughput**: embeddings run on **CPU inside Docker** at ~40
photos/min. There is no CUDA/GPU path in the standard image — the model is
DINOv2 ViT-S/14 and the only acceleration available is Apple **MPS** when
running the backend **natively** on Apple Silicon (much faster than the
containerized CPU run). So "slow" is usually just the CPU baseline for a large
batch.

**Solutions**:

```bash
# Confirm which device the model picked (mps / cuda / cpu)
docker compose exec fastapi python -c \
  "from app.embedding_generator import EmbeddingGenerator as E; print(E()._detect_device())"

# Watch progress — it's working, just CPU-bound
curl -s http://localhost:8000/stats | jq '{completed, pending, failed}'
```

To go faster, run the backend natively on an Apple Silicon Mac (it will use
MPS automatically) instead of in Docker. The job processes ~2 photos
concurrently (a semaphore in `job_queue.py`); raising it rarely helps on CPU
and can starve the event loop.



---



#### 2. "Similarity search is slow for large datasets"



**Causes**:

- Qdrant index is not optimized

- Too many vectors in collection

- Network latency



**Solutions**:



```bash

# Optimize Qdrant index

# Access Qdrant dashboard: http://localhost:6333/dashboard

# Or use Qdrant API to optimize collection



# Increase Qdrant memory limit in docker-compose.yml

services:

  qdrant:

    mem_limit: 4g



# Use pagination to limit results (already supported)
curl -s "http://localhost:8000/similarity-groups?skip=0&limit=50" | jq '.total'

```



---



### Data Issues



#### 1. "Duplicate photos are not being detected"



**Causes**:

- Threshold is too high (>0.95)

- Photos have different metadata (EXIF)

- Embedding model is not sensitive enough



**Solutions**:



```bash

# Lower the threshold

# Try 0.90 or 0.85 for near-duplicates



# Check if photos are actually similar

# View thumbnails in similarity grid



# Verify embeddings are being generated

# Check Qdrant collection size

curl http://localhost:6333/collections/embeddings

```



---



#### 2. "Photos are missing from results"



**Causes**:

- Photos failed to process (check logs)

- Photos are in unsupported format

- Embedding generation failed



**Solutions**:



```bash

# Check backend logs for errors
docker-compose logs fastapi | grep -i error



# Verify photo format is supported
file /path/to/photo.jpg



# Check if photo is corrupted
identify /path/to/photo.jpg  # ImageMagick



# Re-run rescan to retry failed photos

```



---



### Backup & Recovery



#### 1. "Backup failed"



**Solutions**:



```bash

# Check backup manager logs
docker-compose logs fastapi | grep -i backup



# Verify backup directory exists and is writable
docker-compose exec fastapi ls -la /backups



# Manually trigger backup
curl -X POST http://localhost:8000/backup/manual



# Check backup status
curl http://localhost:8000/backup/status

```



---



#### 2. "Recovery failed"



**Solutions**:



```bash

# List available backups
curl http://localhost:8000/backup/status



# Verify backup file exists
docker-compose exec fastapi ls -la /backups/backup-*



# Manually restore from backup

# Stop services
docker-compose down



# Restore database
psql postgresql://postgres:postgres@localhost:5432/app_db < /backups/backup-*.sql



# Restore Qdrant

# Copy backup files to Qdrant storage directory



# Start services
docker-compose up -d

```



---



## Getting Help



If you encounter an issue not covered here:



1. **Check logs**: `docker-compose logs -f backend`

2. **Check health**: `curl http://localhost:8000/health`

3. **Check browser console**: Open DevTools (F12) → Console tab

4. **Search issues**: GitHub Issues or project documentation

5. **Ask for help**: Create a new issue with:

   - Error message and stack trace

   - Steps to reproduce

   - System information (OS, Docker version, etc.)

   - Relevant logs

