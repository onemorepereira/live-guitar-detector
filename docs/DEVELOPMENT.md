# Development

## Local stack

```bash
make dev                                # docker-compose: redis + gateway + worker
cd services/frontend && npm run dev     # vite at http://localhost:5173
```

Vite proxies `/api` and `/ws` to `localhost:8000` (gateway). The frontend
runs on the host (not containerized) for the fastest iteration loop.

The gateway and worker bind-mount their `app/` directories from the host;
edits trigger uvicorn auto-reload (gateway) or require a `docker compose
restart inference-worker` (worker is not hot-reloaded — model load is too
slow to do per-edit).

## Tests

```bash
# Worker
cd services/inference-worker && source .venv/bin/activate && pytest -v

# Gateway
cd services/gateway && source .venv/bin/activate && pytest -v

# Frontend
cd services/frontend && npm test
```

Test markers:

- `requires_model` — Python tests that need the OpenVINO IRs on disk
  (`app/models/`). Auto-skipped if missing.
- `requires_real_fixtures` — accuracy tests that need real guitar photos
  in `services/inference-worker/tests/fixtures/images/`. Auto-skipped
  unless `REAL.txt` exists in that dir.
- `requires_aiortc_peer` — gateway WebRTC negotiation tests. Run with
  `RUN_AIORTC_TESTS=1 pytest`.

## Webcam smoke test (worker only)

The worker has a standalone webcam mode that bypasses the gateway. Useful
for tuning detection thresholds or sanity-checking after a model change.

```bash
cd services/inference-worker
source .venv/bin/activate
python -m app.main --webcam 0
```

A cv2 window opens; press `q` to quit. Per-second latency is printed to
stdout.

## Integration harness

Spins up redis + gateway + worker + a synthetic-frame producer, exercises
the wire path end-to-end, exits 0 on green.

```bash
make test-integration
```

The harness uses synthetic placeholder images — it asserts that detection
events flow, NOT that they classify correctly. For accuracy testing, drop
real photos into `services/inference-worker/tests/fixtures/images/`,
touch `REAL.txt` in that dir, and re-run the worker tests.

## Adding a new guitar model

1. Edit `docs/prompts.md` — add a new entry to the YAML block:

   ```yaml
   - text: "a photograph of a Rickenbacker 4001 bass guitar"
     brand: Rickenbacker
     model: "4001"
   ```

2. (Optional but recommended) add a fixture photo to
   `services/inference-worker/tests/fixtures/images/` and bump the
   accuracy-test threshold in `tests/test_classifier.py`.
3. Update the brand color map in
   `services/frontend/src/components/HUD.tsx` so the HUD has a stroke
   color for the new brand. The default is gray (Unknown).
4. Restart the worker so the text-feature cache is rebuilt:

   ```bash
   docker compose restart inference-worker
   ```

No code change needed for the inference path — the classifier loads
prompts from disk at startup.

## Code style

- Python: ruff + ruff-format, line length 100, target py311.
- TypeScript: strict mode, no `any`, named exports only.
- Pre-commit hooks enforce both — they run automatically on `git commit`.
- Tests use TDD (write failing test → implement → green → commit).
- Conventional Commits: `type(scope): summary`.

See `.pre-commit-config.yaml` and `pyproject.toml` for exact config.

## Project layout

```
services/
  gateway/             FastAPI + aiortc + Redis client + WS forwarder
  inference-worker/    YOLO + ByteTrack + MobileCLIP + voting pipeline
  frontend/            React + Vite + Tailwind + Canvas HUD
deploy/
  helm/guitar-detect/  Helm chart for K3s install
  k3s/                 cluster bootstrap scripts (registry, mkcert, labels)
docs/                  spec, plan, benchmarks, checklists
```
