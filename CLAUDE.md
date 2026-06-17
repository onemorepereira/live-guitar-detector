# CLAUDE.md

Real-time guitar detection: stream a phone camera over WebRTC, detect guitars
with YOLO, classify brand/model with SigLIP/CLIP, smooth with track-based voting.

**`DESIGN.md` is the canonical spec** (§-numbered; kept verbatim — formatters
are configured to skip it). Deep-dives live in `docs/` (DEPLOYMENT, DEVELOPMENT,
CLASSIFIER, TROUBLESHOOTING, BENCHMARKS, E2E_CHECKLIST).

## Architecture

Monorepo of three independently-packaged services under `services/`:

- **gateway** (FastAPI + aiortc) — terminates the browser WebRTC peer,
  rate-limits frames, JPEG-encodes, and `XADD`s them to Redis `frames:{session}`.
  Forwards detection events back over a WebSocket. One session = one browser tab.
- **inference-worker** — consumes `frames:{session}` via Redis consumer groups,
  runs YOLO (OpenVINO) → ByteTrack → SigLIP/CLIP classification → rolling-window
  vote, and `XADD`s events to `detections:{session}`.
- **frontend** (React + Vite + Tailwind) — camera capture, WebRTC, draws
  bboxes + HUD, capture gallery.

Redis is the only cross-service contract (stream wire-format in DESIGN.md
§5.1/§5.3). `CLASSIFIER_MODE` selects `zero_shot` | `probe` | `siglip_probe`
(prod runs `siglip_probe`).

## Commands

`make test` and `make lint` are **stubs** (`echo TODO`) — run per service:

```bash
# Tests — each service has its own venv / node_modules
cd services/gateway          && .venv/bin/pytest
cd services/inference-worker && .venv/bin/pytest          # some tests auto-skip (see markers)
cd services/frontend         && node_modules/.bin/vitest run

# Lint/format — canonical path is pre-commit (ruff + ruff-format + prettier)
pre-commit run --files <changed-files>
cd services/frontend && node_modules/.bin/tsc --noEmit     # frontend typecheck (== npm run lint)

# Local dev stack (podman/docker auto-detected; `make runtime` to check)
make dev                                  # redis + gateway + worker
cd services/frontend && npm run dev       # vite @ :5173, proxies /api + /ws to :8000
```

Python style: ruff, line-length 100, target py311.

## Test markers (auto-skip when prereqs absent)

- `requires_model` — needs the OpenVINO IRs in `app/models/`.
- `requires_real_fixtures` — needs real photos + a `REAL.txt` marker in
  `services/inference-worker/tests/fixtures/images/`.
- `requires_aiortc_peer` — gateway WebRTC negotiation; run with `RUN_AIORTC_TESTS=1`.

## Deploy

Images build from this repo (`make build-images` / `make push-images`); a
separate GitOps repo holds the Helm chart and Flux reconciles. See
`docs/DEPLOYMENT.md`. The `Makefile` `REGISTRY` default is a placeholder —
override `REGISTRY=` for your own registry.

## Conventions

Conventional Commits (`type(scope): description`); branch `type/short-desc`.
No direct commits to `main`. `DESIGN.md` is verbatim — don't let formatters reflow it.
