# Guitar Detect Frontend

React 18 + Vite + Tailwind SPA for the Guitar Detect system.

## Development

```
# Install once
npm install

# Run the dev server (proxies /api and /ws to localhost:8000)
npm run dev

# Run tests
npm test

# Type-check
npm run lint
```

## Manual end-to-end

To exercise the full pipeline:

1. From the repo root: start Redis + gateway + worker (Phase 2 dev compose
   lands in Task 2.16; for now, run each service manually):

   ```
   # Terminal 1 — Redis
   redis-server --port 6379

   # Terminal 2 — gateway (in services/gateway/)
   source .venv/bin/activate
   uvicorn app.main:app --port 8000 --reload

   # Terminal 3 — worker (in services/inference-worker/)
   source .venv/bin/activate
   REDIS_URL=redis://localhost:6379/0 python -m app.main
   ```

2. In another terminal, run the frontend dev server:
   ```
   cd services/frontend
   npm run dev
   ```
3. Open `http://localhost:5173` in your browser (the dev proxy handles
   `/api` and `/ws`).
4. Pick your camera from the dropdown, click **Start**, and point the
   camera at a guitar.
5. A bounding box should appear with `Analyzing…`, then lock onto the
   brand/model within ~2 seconds.

`?debug=1` enables the diagnostics overlay.
