# Guitar Detect

Real-time guitar brand/model detection on a home LAN. Point a phone or
laptop camera at a guitar; the live HUD locks onto the brand and model
within ~2 seconds. CPU-only inference, deployable to a 2-node K3s
cluster.

## What it does

- Detects electric guitars in a webcam stream (YOLOv8n-oiv7).
- Classifies brand + model across 6 targets:
  Gibson Les Paul · SG · Explorer · Flying V · Fender Stratocaster · Telecaster.
- Tracks each guitar across frames; the HUD overlay shows a brand-colored
  bbox with `Brand Model · confidence%`.
- Browser-based; no install on the viewing device beyond a one-time CA trust.

_(screenshot pending — manual gate)_

## Architecture (one paragraph)

Browser captures camera via WebRTC → a FastAPI **gateway** (aiortc decoder)
publishes JPEG frames to **Redis Streams** → an **inference worker**
(YOLOv8n + ByteTrack + MobileCLIP zero-shot classifier) consumes them,
runs a per-track rolling vote for stability, and publishes detection
events back to Redis → gateway forwards events to the browser over
WebSocket, which redraws a canvas overlay at 60 FPS.

See [DESIGN.md](DESIGN.md) for the full spec.

## Quick start (developer laptop)

Prereqs: Python 3.11, Node 20, Docker, ~510 MB of model weights.

```bash
# 1. Download the models (~30 s; one-time)
cd services/inference-worker
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,export]"
python scripts/download_models.py all --out app/models/

# 2. Bring up the dev stack (redis + gateway + worker)
cd ../..
make dev

# 3. In another terminal: frontend dev server
cd services/frontend && npm install && npm run dev

# 4. Open http://localhost:5173 — pick a camera — point at a guitar.
```

Add `?debug=1` to the URL for the diagnostics overlay.

## Docs

- [DEVELOPMENT.md](docs/DEVELOPMENT.md) — local dev loop, testing, prompts editing.
- [DEPLOYMENT.md](docs/DEPLOYMENT.md) — K3s install, Helm chart, upgrade, rollback.
- [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — common issues + fixes.
- [BENCHMARKS.md](docs/BENCHMARKS.md) — Phase 1 latency numbers.
- [E2E_CHECKLIST.md](docs/E2E_CHECKLIST.md) — manual gate procedure.
- [docs/prompts.md](docs/prompts.md) — classifier prompts (edit to retune).
- [docs/plans/](docs/plans/) — the implementation plan that drove the build.

## Status

Built end-to-end from a spec via subagent-driven development; Phase 1
(inference core), Phase 2 (gateway + frontend + integration harness),
Phase 3 (production Docker images), and Phase 4 (K3s Helm chart +
scripts) are complete. Phase 5 added the session gallery, HUD smoothing,
debug panel metrics, and these docs.

| Layer            | Tests                | Status                                  |
| ---------------- | -------------------- | --------------------------------------- |
| Worker (Python)  | 61 passed, 8 skipped | All Phase 1-2 logic covered             |
| Gateway (Python) | 38 passed, 2 skipped | Health, session, redis, WebRTC, WS, API |
| Frontend (TS)    | 58 passed            | Hooks, components, math, integration    |
