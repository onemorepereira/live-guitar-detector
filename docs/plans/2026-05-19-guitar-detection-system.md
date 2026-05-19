# Guitar Detection System — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a real-time, browser-based guitar detection & brand/model classification system per `DESIGN.md`, deployable to a 2-node K3s home cluster.

**Architecture:** Browser captures camera via WebRTC → FastAPI gateway decodes frames via aiortc → Redis Streams bus → inference worker runs YOLOv8n-oiv7 (OpenVINO INT8) + ByteTrack + MobileCLIP zero-shot classifier with per-track rolling vote → detection events flow back via WebSocket → canvas HUD overlay. CPU-only by design; node split is `io` (mobile chip) for gateway/redis and `compute` (Ryzen 7) for inference.

**Tech Stack:** Python 3.11 (FastAPI, aiortc, Ultralytics YOLOv8, OpenCLIP MobileCLIP, OpenVINO 2024.4, Redis 7, loguru, pytest) · TypeScript/React 18 + Vite + Tailwind · Docker · Helm · K3s · Traefik · Longhorn · mkcert.

**Source of truth:** `DESIGN.md` at repo root — read it before each phase. This plan operationalizes it; if they conflict, the design wins unless this plan explicitly amends it.

**Conventions (apply to every task unless overridden):**

- **Commits:** Conventional Commits (`type(scope): summary`), one logical change per commit, no AI attribution.
- **TDD:** Write failing test → run to see it fail → minimal code → run to see it pass → commit. Never the other order.
- **DRY/YAGNI:** No abstractions until the third caller. No code "for later."
- **Python style:** `ruff` + `black` defaults; type hints on all public functions; `pydantic` for any structured data crossing a boundary.
- **TS style:** strict mode on; no `any`; named exports only.
- **No `latest` tags** in Dockerfiles, Helm values, or `requirements*.txt` (pin to exact or `>=x.y,<x.y+1`).
- **Pre-commit:** Add and use `pre-commit` hooks (ruff, black, prettier) from Task 1.4.

---

## Phase 0 — Repo Scaffolding & Tooling

Goal: A working monorepo skeleton with linting, formatting, and pre-commit ready. Nothing functional yet.

### Task 0.1: Top-level scaffolding

**Files:**

- Create: `README.md`
- Create: `Makefile`
- Create: `.env.example`
- Create: `services/.gitkeep`, `deploy/.gitkeep`, `docs/.gitkeep`

**Step 1:** Write `README.md` with a 5-line elevator pitch, link to `DESIGN.md`, link to `docs/plans/`. No screenshots yet.

**Step 2:** Write a `Makefile` with placeholder targets (each just `echo "TODO"` for now):

```
.PHONY: install lint test build-images push-images dev help
help:    ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-20s %s\n", $$1, $$2}'
install: ## Install all dev deps
	@echo "TODO: install"
lint:    ## Run linters across services
	@echo "TODO: lint"
test:    ## Run all unit tests
	@echo "TODO: test"
dev:     ## Run local dev stack (docker-compose)
	@echo "TODO: dev"
build-images: ## Build container images
	@echo "TODO: build-images"
push-images:  ## Push images to local registry
	@echo "TODO: push-images"
```

**Step 3:** Write `.env.example` enumerating every env var from DESIGN.md §5.7 with the documented defaults.

**Step 4:** Commit.

```
git add README.md Makefile .env.example .gitignore DESIGN.md docs/plans services
git commit -m "chore: initial repo scaffolding"
```

### Task 0.2: Pre-commit + Python tooling

**Files:**

- Create: `.pre-commit-config.yaml`
- Create: `pyproject.toml` (root — workspace-level ruff/black config)

**Step 1:** Write `.pre-commit-config.yaml` with: `ruff`, `ruff-format` (or black), `prettier` for `*.md,*.yaml,*.ts,*.tsx`, `end-of-file-fixer`, `trailing-whitespace`, `check-yaml`. Pin all rev hashes.

**Step 2:** Write root `pyproject.toml` with `[tool.ruff]`, `[tool.black]`, `[tool.pytest.ini_options]` defaults shared across services (line length 100, target-version py311).

**Step 3:** Install and run once:

```
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Expected: hooks reformat any noise; rerun to clean.

**Step 4:** Commit.

```
git commit -m "chore: pre-commit + shared python tooling"
```

---

## Phase 1 — Inference Core (CPU, local webcam, no web)

**Phase goal:** A `pipeline.py` runnable as `python -m app.main --webcam 0` that opens the local webcam, draws detection boxes + brand/model labels on a window, and prints rolling latency stats. No Redis, no FastAPI, no Docker yet.

**Phase done-when:**

- Webcam window shows guitars highlighted with stable brand/model labels.
- `scripts/benchmark.py` reports p50 detect+classify+track < 50ms on this host (or document actual numbers — DESIGN.md target is Ryzen 7).
- All unit tests pass: `pytest services/inference-worker/tests -v`.
- Track IDs stable across a 2-second occlusion in manual test.

### Task 1.1: Inference worker package skeleton

**Files:**

- Create: `services/inference-worker/pyproject.toml`
- Create: `services/inference-worker/app/__init__.py`
- Create: `services/inference-worker/app/config.py`
- Create: `services/inference-worker/tests/__init__.py`
- Create: `services/inference-worker/tests/conftest.py`

**Step 1:** Write `pyproject.toml`:

- `[project]` name `guitar-detect-inference-worker`, version `0.1.0`, python `>=3.11,<3.12`
- Deps: `ultralytics>=8.3,<8.4`, `open-clip-torch>=2.24,<3`, `openvino>=2024.4,<2025`, `opencv-python-headless>=4.10,<5`, `numpy>=1.26,<2`, `redis>=5.0,<6`, `pyyaml>=6,<7`, `loguru>=0.7,<1`, `pydantic>=2.6,<3`, `pydantic-settings>=2.2,<3`.
- Dev: `pytest>=8,<9`, `pytest-asyncio>=0.23,<1`, `pytest-cov>=5,<6`, `fakeredis>=2.21,<3`.
- Configure `[tool.setuptools.packages.find]` to find `app`.

**Step 2:** Write `app/config.py` using `pydantic-settings.BaseSettings`. One class `Settings` with fields matching DESIGN.md §5.7 worker table (`REDIS_URL`, `DETECT_CONF`, `DETECT_IOU`, `DETECT_IMGSZ`, `CLIP_INPUT_SIZE`, `VOTE_WINDOW`, `VOTE_STABLE_MIN`, `VOTE_STABLE_CONF`, `MODELS_DIR`, `PROMPTS_FILE`, `OPENVINO_DEVICE`, `OPENVINO_THREADS`). Use exact default values from the design. Add `model_config = SettingsConfigDict(env_file=".env", env_prefix="")`.

**Step 3:** Write `tests/conftest.py` with a `settings` fixture overriding any path defaults to test fixtures dir.

**Step 4:** Install in editable mode and verify import works.

```
cd services/inference-worker && pip install -e ".[dev]"
python -c "from app.config import Settings; print(Settings().model_dump())"
```

Expected: prints a dict with all defaults.

**Step 5:** Commit.

```
git commit -m "feat(worker): package skeleton + Settings"
```

### Task 1.2: Voting module (pure logic, fully TDD)

This task lives entirely in pure-Python land — no ML — so test it exhaustively first.

**Files:**

- Create: `services/inference-worker/app/voting.py`
- Create: `services/inference-worker/tests/test_voting.py`

**Step 1: Write failing tests.** Cover the bullet list from DESIGN.md §5.4 voting:

```python
# tests/test_voting.py
import pytest
from app.voting import TrackVote, SmoothedLabel

def make_label(brand="Gibson", model="Les Paul", conf=0.9):
    return {"brand": brand, "model": model, "confidence": conf}

def test_empty_vote_returns_none_label():
    v = TrackVote(window=15, stable_min=8, stable_conf=0.55)
    out = v.current()
    assert out.label is None
    assert out.stable is False

def test_consistent_label_becomes_stable_after_min_samples():
    v = TrackVote(window=15, stable_min=8, stable_conf=0.55)
    for _ in range(8):
        v.update(make_label("Gibson", "Les Paul", 0.9))
    out = v.current()
    assert out.label == {"brand": "Gibson", "model": "Les Paul"}
    assert out.stable is True
    assert out.confidence > 0.55

def test_below_stable_min_is_not_stable():
    v = TrackVote(window=15, stable_min=8, stable_conf=0.55)
    for _ in range(7):
        v.update(make_label("Gibson", "Les Paul", 0.9))
    assert v.current().stable is False

def test_flapping_labels_not_stable():
    v = TrackVote(window=15, stable_min=8, stable_conf=0.55)
    for i in range(15):
        v.update(make_label("Gibson", "Les Paul", 0.6) if i % 2 else make_label("Fender", "Stratocaster", 0.6))
    assert v.current().stable is False

def test_unknown_winning_emits_null_label_unstable():
    v = TrackVote(window=15, stable_min=8, stable_conf=0.55)
    for _ in range(10):
        v.update(make_label("Unknown", "Unknown", 0.8))
    out = v.current()
    assert out.label is None
    assert out.stable is False

def test_window_eviction():
    v = TrackVote(window=5, stable_min=3, stable_conf=0.5)
    for _ in range(5):
        v.update(make_label("Gibson", "Les Paul", 0.9))
    for _ in range(5):
        v.update(make_label("Fender", "Stratocaster", 0.9))
    out = v.current()
    assert out.label == {"brand": "Fender", "model": "Stratocaster"}

def test_smoothed_confidence_is_ratio_of_winner_to_total():
    v = TrackVote(window=10, stable_min=3, stable_conf=0.5)
    v.update(make_label("Gibson", "Les Paul", 0.8))
    v.update(make_label("Gibson", "Les Paul", 0.8))
    v.update(make_label("Fender", "Stratocaster", 0.4))
    out = v.current()
    # winner weight 1.6 / total 2.0 = 0.8
    assert out.confidence == pytest.approx(0.8, abs=0.01)
```

**Step 2:** Run, expect import errors.

```
pytest services/inference-worker/tests/test_voting.py -v
```

**Step 3:** Implement `voting.py`.

```python
# app/voting.py
from collections import defaultdict, deque
from dataclasses import dataclass

@dataclass(frozen=True)
class SmoothedLabel:
    label: dict | None      # {"brand": str, "model": str} or None
    confidence: float       # 0..1
    stable: bool
    samples: int

class TrackVote:
    def __init__(self, window: int, stable_min: int, stable_conf: float):
        self._buf: deque[tuple[str, str, float]] = deque(maxlen=window)
        self._stable_min = stable_min
        self._stable_conf = stable_conf

    def update(self, raw: dict) -> None:
        self._buf.append((raw["brand"], raw["model"], float(raw["confidence"])))

    def current(self) -> SmoothedLabel:
        if not self._buf:
            return SmoothedLabel(None, 0.0, False, 0)
        weights: dict[tuple[str, str], float] = defaultdict(float)
        total = 0.0
        for brand, model, conf in self._buf:
            weights[(brand, model)] += conf
            total += conf
        winner = max(weights.items(), key=lambda kv: kv[1])
        (brand, model), w = winner
        smoothed_conf = w / total if total > 0 else 0.0
        is_unknown = brand == "Unknown" or model == "Unknown"
        label = None if is_unknown else {"brand": brand, "model": model}
        stable = (
            len(self._buf) >= self._stable_min
            and smoothed_conf >= self._stable_conf
            and not is_unknown
        )
        return SmoothedLabel(label, smoothed_conf, stable, len(self._buf))
```

**Step 4:** Run, expect all tests pass.

```
pytest services/inference-worker/tests/test_voting.py -v
```

**Step 5:** Commit.

```
git commit -m "feat(worker): rolling vote with stability threshold"
```

### Task 1.3: Track lifecycle manager

The voting module is per-track; we also need to prune tracks not seen for 90 frames (DESIGN.md §5.4) and decide _when_ to classify.

**Files:**

- Create: `services/inference-worker/app/tracks.py`
- Create: `services/inference-worker/tests/test_tracks.py`

**Step 1:** Write failing tests for:

- `TrackRegistry.observe(track_id, frame_no)` records a sighting.
- `TrackRegistry.should_classify(track_id, frame_no, stable: bool)` returns True for first 5 frames of a new track, every 6th frame for unstable, every 30th for stable. Returns False if track bbox area is below threshold (pass area in).
- `TrackRegistry.prune(current_frame_no)` removes tracks not seen in 90 frames; returns list of pruned IDs.
- `TrackRegistry.age(track_id, frame_no)` returns frames since first sighting.

(Write each test as its own function. Include edge cases: brand-new track returns True from `should_classify`; track at frame 30 returns True only when policy says so.)

**Step 2:** Run, see failures.

**Step 3:** Implement minimally. Keep state in dicts; no premature abstraction.

**Step 4:** Run, expect all pass.

**Step 5:** Commit `feat(worker): track lifecycle + classify scheduling`.

### Task 1.4: Prompts file + loader

**Files:**

- Create: `docs/prompts.md` (verbatim from DESIGN.md §5.4 YAML block)
- Create: `services/inference-worker/app/prompts.py`
- Create: `services/inference-worker/tests/test_prompts.py`
- Create: `services/inference-worker/tests/fixtures/prompts_minimal.yaml`

**Step 1:** Copy the prompts YAML block from DESIGN.md §5.4 verbatim into `docs/prompts.md`. The loader should read the YAML directly out of the `.md` file (single ```yaml fenced block extracted).

**Step 2:** Write failing tests:

- `load_prompts(path)` returns list of `{"text", "brand", "model"}` dicts.
- All 6 target (brand, model) pairs from DESIGN.md §1.2 present.
- Three `Unknown`-labelled rejection prompts present.
- Fixture `prompts_minimal.yaml` with 2 prompts loads correctly.
- Malformed YAML raises `ValueError` with a clear message.

**Step 3:** Implement `prompts.py`:

- Function `load_prompts(path: Path) -> list[Prompt]` (Prompt is a `pydantic.BaseModel` with text/brand/model).
- If path ends with `.md`, extract the first ```yaml block; else parse the file as YAML.
- Validate each entry; raise on missing fields.

**Step 4:** Pass. Commit `feat(worker): prompts loader (YAML in Markdown)`.

### Task 1.5: Model download + OpenVINO export script

This task is heavy on infra (downloads weights, exports them). Test it minimally — most validation is "the file exists with the right name."

**Files:**

- Create: `services/inference-worker/scripts/download_models.py`
- Create: `services/inference-worker/scripts/README.md`

**Step 1:** Write `download_models.py` as a CLI (argparse) with subcommands:

- `download-yolo --out DIR` → uses `ultralytics.YOLO('yolov8n-oiv7.pt')` to fetch; saves `.pt` to `DIR`.
- `export-yolo --in PATH --out DIR` → `YOLO(PATH).export(format='openvino', int8=True, data='coco128.yaml')` — note: Ultralytics' `int8=True` needs a calibration dataset; if `coco128.yaml` doesn't satisfy OIv7 classes, fall back to `int8=False` for now and log a clear WARNING that quantization was skipped.
- `download-clip --out DIR` → uses `open_clip.create_model_and_transforms('MobileCLIP-S0', pretrained='datacompdr')` and saves state dict.
- `export-clip --in PATH --out DIR` → exports both image and text towers to ONNX, then `ovc` (OpenVINO Model Converter) to IR. INT8 image tower; FP16 text tower (per DESIGN.md §5.4).
- `all --out DIR` → runs every step in sequence.

**Step 2:** Write `scripts/README.md` documenting:

- Disk needed (~2GB)
- Time needed (~5-10 min first run)
- Expected output tree (`yolov8n-oiv7-int8/`, `mobileclip-s0-image-int8/`, `mobileclip-s0-text-fp16/`)
- How to run: `python scripts/download_models.py all --out app/models/`

**Step 3:** Run the script end-to-end:

```
cd services/inference-worker
python scripts/download_models.py all --out app/models/
```

Expected: model files appear; log lines for each step; total < 10 min.

**Step 4:** Verify presence; do **not** commit the model files (they're in `.gitignore`).

```
ls -la services/inference-worker/app/models/
```

Expected: yolov8n-oiv7 OpenVINO IR (.xml + .bin), mobileclip image + text IRs.

**Step 5:** Commit the script + README.

```
git commit -m "feat(worker): model download + OpenVINO export script"
```

> **If INT8 quantization fails for YOLO due to missing calibration data:** document the failure in `scripts/README.md`, ship with FP32/FP16 first, and open a follow-up task. Do not block Phase 1 on quantization.

### Task 1.6: Detector wrapper

**Files:**

- Create: `services/inference-worker/app/detector.py`
- Create: `services/inference-worker/tests/test_detector.py`
- Create: `services/inference-worker/tests/fixtures/images/` (commit ~6 test JPEGs: 2 Strats, 2 Les Pauls, 1 acoustic, 1 no-guitar street scene). Source: any public-domain or self-photographed images. Names: `strat_01.jpg`, `lp_01.jpg`, etc.

**Step 1:** Write failing tests:

- `Detector(model_dir, conf=0.35, iou=0.5, imgsz=416)` constructs.
- `detector.detect(frame)` on `street_scene.jpg` returns empty list.
- `detector.detect(frame)` on `lp_01.jpg` returns ≥ 1 detection, all with `class_name == "Guitar"`, `conf >= 0.35`, bbox within frame bounds.
- `detector.detect_and_track(frame)` on a sequence of 10 frames of `lp_01.jpg` returns the same `track_id` across all 10 frames.
- Detections returned as a list of `Detection` pydantic models with fields `track_id: int | None`, `bbox_xyxy: tuple[float,float,float,float]`, `confidence: float`, `class_name: str`.

**Step 2:** Run, see failures.

**Step 3:** Implement `detector.py`:

- Loads OpenVINO YOLO via `ultralytics.YOLO(openvino_xml_path, task='detect')`.
- At load time, scans `model.names`, asserts `"Guitar"` is present, stores the class ID; raises `RuntimeError` if missing (fail-fast per DESIGN.md §5.4).
- `detect(frame)` calls `model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, classes=[guitar_id], verbose=False)`; parses result.
- `detect_and_track(frame)` calls `model.track(frame, persist=True, tracker='bytetrack.yaml', imgsz=..., conf=..., iou=..., classes=[guitar_id], verbose=False)`.
- Mark tests requiring the actual model with `@pytest.mark.requires_model` and skip when `MODELS_DIR` is unset or empty.

**Step 4:** Run with models present, expect pass. Without models, expect skips.

**Step 5:** Commit `feat(worker): YOLO+ByteTrack detector wrapper`.

### Task 1.7: Classifier wrapper

**Files:**

- Create: `services/inference-worker/app/classifier.py`
- Create: `services/inference-worker/tests/test_classifier.py`

**Step 1:** Write failing tests:

- `Classifier(model_dir, prompts, input_size=224)` constructs; precomputes text features once.
- `classifier.classify(image_bgr)` returns a dict `{"brand", "model", "confidence"}` with non-empty values and `0 <= confidence <= 1`.
- Accuracy fixture test: feed all 6 brand+model fixture images, assert ≥ 5/6 classify correctly (top-1).
- Acoustic fixture: `classify(acoustic.jpg)` returns `brand == "Unknown" OR model == "Unknown"` (the rejection prompts won).
- Mark with `@pytest.mark.requires_model`.

**Step 2:** Run, see failures.

**Step 3:** Implement per DESIGN.md §5.4 classifier:

- Load OpenVINO image + text towers via `openvino.Core().compile_model(...)`.
- At init, run text tower once over all prompts; cache the resulting feature tensor.
- `classify(image_bgr)`:
  - Convert BGR→RGB, pad to square (replicate-edge or black-pad — match what MobileCLIP preprocessing expects; check the open_clip transforms for the chosen variant), resize to `input_size`.
  - Apply mean/std normalization (use `open_clip.OPENAI_CLIP_MEAN/STD` or the model's specific values).
  - Run image tower.
  - Cosine sim against cached text features; multiply by temperature 100; softmax → probabilities.
  - Take argmax; map to `prompts[idx]`; return `{"brand", "model", "confidence": prob}`.

**Step 4:** Run, expect ≥5/6 fixture pass.

**Step 5:** If accuracy fails: iterate on prompts in `docs/prompts.md` only (no code change). Document any prompt changes in the commit message.

**Step 6:** Commit `feat(worker): MobileCLIP zero-shot classifier`.

### Task 1.8: Pipeline orchestrator

**Files:**

- Create: `services/inference-worker/app/pipeline.py`
- Create: `services/inference-worker/tests/test_pipeline.py`

**Step 1:** Write failing test:

- `Pipeline(detector, classifier, settings).process_frame(frame, frame_no)` returns a `DetectionEvent`-shaped dict matching DESIGN.md §5.1, with `tracks` populated, `inference_ts` set.
- Across 20 frames of the same fixture, the same track ID converges to a stable label by frame 15 (the vote window).
- For a frame with no guitars, returns event with empty `tracks` list.

**Step 2:** Run, see failures.

**Step 3:** Implement per DESIGN.md §5.4 pipeline pseudocode. Use the `Detector`, `Classifier`, `TrackRegistry`, and a `dict[int, TrackVote]` keyed by track ID. Bbox normalization (pixel → 0..1) happens here, not in `Detector`. `inference_ts = int(time.time() * 1000)`.

**Step 4:** Run, expect pass.

**Step 5:** Commit `feat(worker): end-to-end pipeline orchestration`.

### Task 1.9: Webcam runner + on-screen overlay (manual gate)

**Files:**

- Create: `services/inference-worker/app/main.py`
- Create: `services/inference-worker/scripts/webcam_demo.py`

**Step 1:** Write `scripts/webcam_demo.py`: opens `cv2.VideoCapture(args.cam)`, instantiates `Pipeline`, loop reads frames, calls `process_frame`, draws boxes + labels on the frame, shows window. Press `q` to quit. Prints per-second average latency.

**Step 2:** Write `app/main.py`: thin CLI entry that either runs `webcam_demo` (`--webcam IDX`) or — in later phases — the Redis consumer loop (placeholder for Phase 2.x). For Phase 1, only `--webcam` is implemented; the Redis path raises `NotImplementedError`.

**Step 3:** Manual test:

```
cd services/inference-worker
python -m app.main --webcam 0
```

Expected: window opens, point camera at a guitar (printed photo OK), see brand/model lock on within ~2s. Press `q` to quit.

**Step 4:** Document the manual procedure briefly in `services/inference-worker/README.md`.

**Step 5:** Commit `feat(worker): webcam demo runner`.

### Task 1.10: Benchmark script

**Files:**

- Create: `services/inference-worker/scripts/benchmark.py`

**Step 1:** Write a script that:

- Loads `Pipeline`.
- Replays N=600 fixture frames (cycle through fixtures) at a target FPS.
- Records per-stage timings (detect_ms, classify_ms, vote_ms, total_ms).
- Prints p50/p95/p99 for each stage at the end. Reports dropped frames (frames where total > 1/target_fps).

**Step 2:** Run on this host:

```
python scripts/benchmark.py --frames 600 --target-fps 15
```

Capture the output. Add it to a new section of `docs/BENCHMARKS.md` with the date and host description.

**Step 3:** Phase-1 gate: p50 total_ms < 50ms. **If above**, do NOT proceed to Phase 2 without explicit user sign-off — pick from DESIGN.md §10.5 tuning checklist (drop imgsz to 320, etc.) and re-benchmark.

**Step 4:** Commit `chore(worker): benchmark script + initial numbers`.

### Phase 1 wrap-up

- All Phase 1 unit tests green: `pytest services/inference-worker -v`
- Manual webcam demo works
- Benchmark recorded
- **Commit a tag:** `git tag phase-1-done`

---

## Phase 2 — Gateway + Frontend (single-host docker-compose)

**Phase goal:** Browser → camera → server → overlay end-to-end on a developer laptop. `docker-compose up` brings gateway + worker + Redis; `https://localhost:8000` shows the app.

**Phase done-when:**

- Open `https://localhost:8000` on host browser; pick camera; see HUD overlay locking onto guitars within 2s.
- All unit + integration tests green.
- E2E latency < 200ms p95 measured via the debug overlay.

### Task 2.1: Gateway package skeleton

**Files:**

- Create: `services/gateway/pyproject.toml`
- Create: `services/gateway/app/__init__.py`
- Create: `services/gateway/app/config.py`
- Create: `services/gateway/app/main.py`
- Create: `services/gateway/tests/__init__.py`
- Create: `services/gateway/tests/conftest.py`

**Step 1:** Write `pyproject.toml`: deps `fastapi>=0.110,<0.120`, `uvicorn[standard]>=0.27,<0.40`, `aiortc>=1.9,<2`, `av>=11,<13`, `redis>=5.0,<6`, `pydantic>=2.6,<3`, `pydantic-settings>=2.2,<3`, `loguru>=0.7,<1`, `opencv-python-headless>=4.10,<5`, `numpy>=1.26,<2`. Dev: `pytest>=8,<9`, `httpx>=0.27,<1`, `pytest-asyncio>=0.23,<1`, `fakeredis>=2.21,<3`.

**Step 2:** `app/config.py` mirrors gateway env vars from DESIGN.md §5.7.

**Step 3:** `app/main.py` minimal: FastAPI app with `/healthz` returning `{"ok": true}` and `/readyz` returning 200 only if a `redis.ping()` succeeds.

**Step 4:** Run: `cd services/gateway && pip install -e ".[dev]" && uvicorn app.main:app --port 8000` → `curl localhost:8000/healthz`. Expected 200.

**Step 5:** Commit `feat(gateway): FastAPI skeleton + health endpoints`.

### Task 2.2: Session module (TDD)

**Files:**

- Create: `services/gateway/app/session.py`
- Create: `services/gateway/tests/test_session.py`

**Step 1:** Tests (use `fakeredis`):

- `SessionManager.create(session_id)` initializes Redis streams (publishes nothing yet — but `XADD` with `NOMKSTREAM` would fail later, so use `XADD` with `*` and immediately trim, or rely on first publish creating the stream — pick one, document the choice in a code comment).
- `create` then `create` again → `SessionAlreadyExists`.
- `delete(session_id)` removes streams + session hash + removes from `sessions:active` set.
- `touch(session_id)` updates `last_frame_ts`.
- `idle_sessions(timeout_s)` returns session_ids where `last_frame_ts` is older than `timeout_s`.
- Concurrent `create` from two coroutines for the same id — exactly one wins. (Use `SET ... NX`.)

**Step 2:** Run, see failures.

**Step 3:** Implement `session.py` using `redis.asyncio`. Use `SET session:{id} <json> NX EX 60` for the create-once guarantee. `sessions:active` is a Redis Set. Make all methods `async`.

**Step 4:** Run, expect pass.

**Step 5:** Commit `feat(gateway): session lifecycle on Redis`.

### Task 2.3: Redis I/O helpers (TDD)

**Files:**

- Create: `services/gateway/app/redis_io.py`
- Create: `services/gateway/tests/test_redis_io.py`

**Step 1:** Tests:

- `publish_frame(r, session_id, jpeg_bytes, frame_id, frame_ts, w, h)` calls `XADD frames:{session_id} MAXLEN ~ 30 ...`.
- `consume_detections(r, session_id)` is an async generator yielding events as they arrive; integrates against `fakeredis` with `XADD` pumping side-channel.
- Round-trip: publish 5 frames, consume on a separate task, all 5 are received in order.

**Step 2-4:** Standard TDD loop.

**Step 5:** Commit `feat(gateway): redis stream I/O helpers`.

### Task 2.4: WebRTC peer + frame ingest

**Files:**

- Create: `services/gateway/app/webrtc.py`
- Create: `services/gateway/tests/test_webrtc.py`

**Step 1:** Tests (these are integration-heavy; use a real aiortc test peer):

- `WebRTCManager.handle_offer(session_id, sdp)` returns a valid answer SDP.
- Posting a video track to the peer triggers frame-receive callbacks that publish to `fakeredis`.
- Peer state transition `closed` → triggers `SessionManager.delete`.
- Skip integration tests with `@pytest.mark.requires_aiortc_peer` if test harness is too flaky; keep at least the SDP-shape test.

**Step 2-4:** Standard loop. Implementation must include the **30 FPS ingest rate limit** (drop frames if `now - last_published_ts < 33ms`) from DESIGN.md §5.5.

**Step 5:** Commit `feat(gateway): WebRTC peer with frame ingest`.

### Task 2.5: WebSocket detection forwarder

**Files:**

- Create: `services/gateway/app/websocket.py`
- Create: `services/gateway/tests/test_websocket.py`

**Step 1:** Tests using FastAPI TestClient WebSocket support:

- Connect WS with `session_id=foo`; publish a `DetectionEvent` to `detections:foo` in Redis; assert the test client receives it as JSON.
- Ping/pong: client sends `{"type":"ping"}`, receives `{"type":"pong"}`.
- WS close → forwarder task exits and (combined with idle sweep) cleans the session.

**Step 2-4:** Standard loop.

**Step 5:** Commit `feat(gateway): WS detection event forwarder`.

### Task 2.6: API wiring

**Files:**

- Modify: `services/gateway/app/main.py`
- Create: `services/gateway/tests/test_api.py`

**Step 1:** Add the four API endpoints from DESIGN.md §5.2 (`POST /api/session`, `DELETE /api/session/{id}`, `POST /api/webrtc/offer`, `WS /ws`). Plumb through `SessionManager`, `WebRTCManager`, `WebSocket` forwarder.

**Step 2:** Add request/response pydantic models in `app/models.py` matching §5.1.

**Step 3:** Tests via `httpx.AsyncClient` (FastAPI test client) for HTTP endpoints; reuse the WS test for the WS path. Cover 200, 409, 422 (validation), 404 (delete missing).

**Step 4:** Add a background task on app startup that runs `idle_sessions` every 2s and tears down sessions exceeding `SESSION_IDLE_TIMEOUT_S`.

**Step 5:** Run all gateway tests green.

**Step 6:** Commit `feat(gateway): wire session, WebRTC, WS endpoints`.

### Task 2.7: Worker — Redis consumer mode

This is the second consumer of the same `Pipeline` from Phase 1. Most code already exists; this task wires it to Redis.

**Files:**

- Modify: `services/inference-worker/app/main.py`
- Create: `services/inference-worker/app/consumer.py`
- Create: `services/inference-worker/tests/test_consumer.py`

**Step 1:** Tests (with `fakeredis`):

- Worker registers in consumer group `inference` on `frames:<id>` when `<id>` is in `sessions:active`.
- Frames published to `frames:<id>` are consumed and turned into events on `detections:<id>`.
- Removing the session from `sessions:active` stops consumption for that session.
- ACK is sent on success; on classifier exception, message is XREADGROUP-claimed back after timeout but NOT silently dropped.

**Step 2-4:** Standard loop. Use `redis.asyncio` and the same pipeline class from Phase 1. Discovery loop runs every 1s per DESIGN.md §5.3.

**Step 5:** Update `app/main.py` so `python -m app.main` (no args) enters Redis consumer mode; `--webcam` keeps the old behavior.

**Step 6:** Commit `feat(worker): redis consumer mode`.

### Task 2.8: Frontend scaffolding

**Files:**

- Create: `services/frontend/package.json`
- Create: `services/frontend/vite.config.ts`
- Create: `services/frontend/tsconfig.json`
- Create: `services/frontend/tailwind.config.js`
- Create: `services/frontend/postcss.config.js`
- Create: `services/frontend/index.html`
- Create: `services/frontend/src/main.tsx`
- Create: `services/frontend/src/App.tsx`
- Create: `services/frontend/src/styles/index.css`
- Create: `services/frontend/tests/setup.ts`

**Step 1:** Use `npm create vite@latest frontend -- --template react-ts` as the _reference_, but write the files by hand so versions are pinned (`react@18.3.x`, `vite@5.x`, `tailwindcss@3.4.x`, `typescript@5.5.x`). Add `vitest@2.x` for tests.

**Step 2:** Configure Tailwind. Initialize `src/App.tsx` with a placeholder `<h1>Guitar Detect</h1>`.

**Step 3:** Verify: `cd services/frontend && npm install && npm run dev` opens at `http://localhost:5173`.

**Step 4:** Commit `feat(frontend): vite + react + tailwind scaffolding`.

### Task 2.9: Types & utility — denormalizeBbox

**Files:**

- Create: `services/frontend/src/types/detection.ts`
- Create: `services/frontend/src/lib/bbox.ts`
- Create: `services/frontend/src/lib/bbox.test.ts`

**Step 1:** Define the TS types from DESIGN.md §5.1 (`DetectionEvent`, `TrackDetection`, `ClassificationLabel`) in `types/detection.ts`. Use string literal unions for brand/model exactly as in the spec.

**Step 2:** Write `bbox.test.ts` with vitest:

- `denormalizeBbox([0, 0, 1, 1], {videoW: 1920, videoH: 1080, elW: 960, elH: 540})` → `[0, 0, 960, 540]`.
- Letterbox case: video 1920×1080 in a 1000×1000 element (object-fit: contain) → bbox shifted vertically by the letterbox margins.
- Pillarbox case: video 1080×1920 in a 1000×1000 element → bbox shifted horizontally.
- Edge bbox at `[0,0,1,1]` always covers the visible video area, not the letterbox bars.

**Step 3:** Implement `denormalizeBbox` carefully. Write the letterbox math explicitly (`scale = min(elW/videoW, elH/videoH); offsetX = (elW - videoW*scale)/2`...).

**Step 4:** `npx vitest run` — all pass.

**Step 5:** Commit `feat(frontend): types + bbox denormalization`.

### Task 2.10: useCamera hook

**Files:**

- Create: `services/frontend/src/hooks/useCamera.ts`
- Create: `services/frontend/src/hooks/useCamera.test.ts`

**Step 1:** Tests (mock `navigator.mediaDevices`):

- `useCamera()` calls `enumerateDevices` and returns videoinput devices.
- Selecting a device id calls `getUserMedia({video: {deviceId: {exact}}})`.
- Permission denial surfaces as an error state, not a thrown.

**Step 2-4:** Standard loop. Hook returns `{devices, selected, stream, select, error}`.

**Step 5:** Commit `feat(frontend): useCamera hook`.

### Task 2.11: useWebRTC hook

**Files:**

- Create: `services/frontend/src/hooks/useWebRTC.ts`
- Create: `services/frontend/src/api/webrtc.ts`

**Step 1:** Write `api/webrtc.ts`: `postOffer(sessionId, sdp)` calls `POST /api/webrtc/offer`.

**Step 2:** Write `useWebRTC(stream, sessionId)`:

- Creates `RTCPeerConnection`, adds tracks, creates offer.
- Calls `postOffer`, sets remote description.
- Returns `{state, error}`.

**Step 3:** Test minimally via component test (mocking RTCPeerConnection is tedious — the integration test will cover this in Task 2.16). Smoke-test that the hook constructs without error.

**Step 4:** Commit `feat(frontend): useWebRTC hook`.

### Task 2.12: useDetections hook

**Files:**

- Create: `services/frontend/src/hooks/useDetections.ts`
- Create: `services/frontend/src/hooks/useDetections.test.ts`

**Step 1:** Tests (use `mock-socket` or vitest mocks for WebSocket):

- Opens WS at `/ws?session_id=...`.
- Parses incoming JSON; only the most recent event is exposed via the hook (we don't queue stale frames).
- Auto-reconnects with exponential backoff capped at 5s.
- Sends `{"type":"ping"}` every 5s; logs warning if no pong within 3s.

**Step 2-4:** Standard loop.

**Step 5:** Commit `feat(frontend): useDetections hook with reconnect`.

### Task 2.13: HUD component

**Files:**

- Create: `services/frontend/src/components/HUD.tsx`
- Create: `services/frontend/src/components/HUD.test.tsx`

**Step 1:** Tests via mocked canvas 2D context (vitest + `vitest-canvas-mock` or hand-rolled spy):

- For a stable Gibson Les Paul track, `strokeStyle` is set to `#C8A45C`.
- For a Fender track, `#F5F5F5`.
- For `stable: false`, the label reads `Analyzing…` (italic).
- For a label near the top of the frame, text renders below the box rather than above.
- Opacity ramps from 0.3 to 1.0 across `age_frames` 1→5.

**Step 2-4:** Standard loop. Use `requestAnimationFrame` for rendering; the component takes `tracks` and `videoRect` as props.

**Step 5:** Commit `feat(frontend): HUD canvas overlay`.

### Task 2.14: CameraPicker + VideoStage + DebugPanel

**Files:**

- Create: `services/frontend/src/components/CameraPicker.tsx`
- Create: `services/frontend/src/components/VideoStage.tsx`
- Create: `services/frontend/src/components/DebugPanel.tsx`

**Step 1:** `CameraPicker`: dropdown bound to `useCamera`. Disables Start until camera selected.

**Step 2:** `VideoStage`: `<video>` (autoplay, muted, playsinline) overlaid with `<canvas>`; size synced via `ResizeObserver`. Renders `HUD` inside.

**Step 3:** `DebugPanel`: only renders if `URLSearchParams.has('debug')`. Shows FPS, ping latency, WS state, last frame_ts age.

**Step 4:** No unit tests required for these (mostly composition); manual gate covers them.

**Step 5:** Commit `feat(frontend): camera picker, video stage, debug panel`.

### Task 2.15: App composition + manual E2E

**Files:**

- Modify: `services/frontend/src/App.tsx`
- Create: `services/frontend/src/api/session.ts`

**Step 1:** `api/session.ts`: `createSession(id)`, `deleteSession(id)`.

**Step 2:** Wire `App.tsx`:

- Landing state: shows `CameraPicker`, "Start" button.
- On Start: `crypto.randomUUID()`, `createSession`, `useDetections`, `useWebRTC`, render `VideoStage`.
- "Stop" button: `deleteSession`, reset state.

**Step 3:** Configure Vite proxy in `vite.config.ts` so `/api` and `/ws` route to the gateway during dev.

**Step 4:** **Manual gate** — start gateway + worker + redis + frontend dev server, open `https://localhost:5173`, run the full flow. Iterate until it works end-to-end on a printed photo of a Strat and Les Paul.

**Step 5:** Commit `feat(frontend): wire end-to-end app flow`.

### Task 2.16: docker-compose for dev

**Files:**

- Create: `docker-compose.yml`
- Create: `services/gateway/Dockerfile.dev`
- Create: `services/inference-worker/Dockerfile.dev`

**Step 1:** Write a dev docker-compose:

- `redis` (redis:7.2-alpine) on internal network.
- `gateway` built from `Dockerfile.dev` with `services/gateway` bind-mounted, hot-reload via `uvicorn --reload`.
- `inference-worker` similar with bind-mount + `MODELS_DIR=/models` and a host bind-mount for downloaded models.
- Frontend NOT containerized in dev — run with `npm run dev` on host (simpler iteration). Vite proxy to gateway.

**Step 2:** Add Makefile targets: `make dev` → `docker compose up --build`.

**Step 3:** Manual gate again: `make dev` + `npm run dev` (frontend) + browser → end-to-end works.

**Step 4:** Commit `chore: dev docker-compose stack`.

### Task 2.17: Integration test harness (synthetic frame producer)

**Files:**

- Create: `docker-compose.test.yml`
- Create: `services/inference-worker/tests/integration/`
- Create: `services/inference-worker/tests/integration/synthetic_producer.py`
- Create: `services/inference-worker/tests/integration/assert_subscriber.py`

**Step 1:** Implement per DESIGN.md §9.3:

- `synthetic_producer`: POSTs `/api/session`, opens WS, then `XADD frames:{id} ...` with fixture JPEGs at 15 FPS for 30s.
- `assert_subscriber`: subscribes to the same WS, asserts that detection events arrive, track IDs stable, stable label emerges within 1s of feeding a clear fixture; exits 0 on success.

**Step 2:** Add `make test-integration` that brings up the test compose and runs the assertions.

**Step 3:** Run; debug until clean.

**Step 4:** Commit `test: integration harness with synthetic frames`.

### Phase 2 wrap-up

- All unit + integration tests green.
- Manual E2E in browser works (host + printed-photo test).
- Tag: `git tag phase-2-done`.

---

## Phase 3 — Containerization (production-shaped images)

**Phase goal:** Final Dockerfiles produce images that match DESIGN.md §6 size targets, with the frontend baked into the gateway image.

**Phase done-when:**

- `make build-images` produces `gateway` (< 350MB) and `inference-worker` (< 1.5GB).
- `docker compose -f docker-compose.yml up` (without bind-mounts) reproduces Phase 2's E2E behavior.
- Health checks pass on cold start within 30s.

### Task 3.1: Frontend Dockerfile (multi-stage build → static assets)

**Files:**

- Create: `services/frontend/Dockerfile`

**Step 1:** Multi-stage:

- Stage `build`: `node:20.11-alpine`, `npm ci`, `npm run build` → `/app/dist`.
- Stage `export`: scratch with `COPY --from=build /app/dist /dist`. (Used as build context, not actually run.)

This image is consumed by the gateway image in the next task — it is not a runtime image.

**Step 2:** Build it: `docker build services/frontend -t guitar-detect/frontend-assets:dev`.

**Step 3:** Commit `feat(frontend): production Dockerfile (static assets)`.

### Task 3.2: Gateway production Dockerfile

**Files:**

- Create: `services/gateway/Dockerfile`

**Step 1:** Multi-stage:

- Stage `frontend`: `FROM guitar-detect/frontend-assets:dev as frontend` (or `ARG FRONTEND_IMAGE` for flexibility).
- Stage `python-build`: `python:3.11-slim`, install `pip-tools`, generate locked deps, install into `/install`.
- Stage `runtime`: `python:3.11-slim`. Install runtime system deps for `aiortc`: `libavcodec59`, `libavformat59`, `libavutil57`, `libswscale6`, `libsrtp2-1`, `libopus0`, `libvpx7`. `COPY --from=python-build /install /usr/local`. `COPY --from=frontend /dist /app/static`. `COPY services/gateway/app /app/app`. Entrypoint: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1`.

**Step 2:** Add static-file serving to `app/main.py`: `app.mount("/", StaticFiles(directory="static", html=True), name="static")` — but only after API routes are registered so they take precedence.

**Step 3:** Build and inspect size: `docker build services/gateway -t guitar-detect/gateway:dev && docker images | grep gateway`. Expected < 350MB; if over, prune apt cache / `--no-install-recommends` / multi-stage cleanup.

**Step 4:** Commit `feat(gateway): production Dockerfile`.

### Task 3.3: Inference worker production Dockerfile

**Files:**

- Modify: `services/inference-worker/Dockerfile` (was `.dev`; this is the prod sibling — keep both).
- Create: `services/inference-worker/Dockerfile`

**Step 1:** Base: `openvino/ubuntu22_runtime:2024.4.0`. Install `python3-pip`. `pip install` worker deps. Copy `app/` and `scripts/`. Run `scripts/download_models.py all --out /models` during build (cache layer). Entrypoint: `python -m app.main`.

**Step 2:** Build and inspect: target < 1.5GB. The model download is the slowest layer — order it for cache reuse.

**Step 3:** Commit `feat(worker): production Dockerfile with baked models`.

### Task 3.4: Production docker-compose

**Files:**

- Modify: `docker-compose.yml` to add a `profile: prod` section using the production images and no bind-mounts.

**Step 1:** Add profile. `make prod` runs `docker compose --profile prod up`.

**Step 2:** Manual E2E with prod images. Latency budget met.

**Step 3:** Commit `chore: production docker-compose profile`.

### Phase 3 wrap-up

- `git tag phase-3-done`.

---

## Phase 4 — K3s Deployment

**Phase goal:** Running on the home cluster at `https://guitars.home.lan`.

**Phase done-when:**

- Phone and desktop on LAN can use the app over HTTPS.
- Pods scheduled on the expected nodes (`kubectl get pods -o wide` confirms).
- Killing the inference pod mid-session recovers within 10s.

### Task 4.1: Local registry deployment

**Files:**

- Create: `deploy/k3s/install-registry.sh`
- Modify: `Makefile` to add `push-images` target (`docker tag` + `docker push registry.local:5000/...`).

**Step 1:** Script deploys `registry:2.8` as a Deployment+Service in the `kube-system` namespace (or a dedicated `registry` namespace), backed by a Longhorn PVC. Includes the `/etc/rancher/k3s/registries.yaml` snippet to add to each node, with `chmod 644` and a `systemctl restart k3s` reminder printed at the end.

**Step 2:** Run the script on the cluster. Verify with `curl http://registry.local:5000/v2/`.

**Step 3:** Push Phase 3 images: `make push-images TAG=0.1.0`.

**Step 4:** Commit `feat(deploy): local registry install + push script`.

### Task 4.2: mkcert setup

**Files:**

- Create: `deploy/k3s/install-mkcert-cert.sh`
- Create: `deploy/k3s/README.md`

**Step 1:** Script:

- Runs `mkcert -install` (idempotent).
- Generates `mkcert guitars.home.lan` → `guitars.home.lan.pem` + `guitars.home.lan-key.pem`.
- Creates k8s secret: `kubectl create secret tls guitars-tls --cert=... --key=... -n guitar-detect --dry-run=client -o yaml | kubectl apply -f -`.

**Step 2:** Document in `deploy/k3s/README.md`:

- How to copy the mkcert root CA (`mkcert -CAROOT`) to each viewing device (Android, iOS, macOS, Linux Chrome) with platform-specific commands.
- How to add `guitars.home.lan` to router DNS or device `/etc/hosts`.

**Step 3:** Run the script; verify secret exists.

**Step 4:** Commit `feat(deploy): mkcert + TLS secret install`.

### Task 4.3: Node labeling script

**Files:**

- Create: `deploy/k3s/label-nodes.sh`

**Step 1:** Script takes node names as args or reads `MOBILE_NODE`/`COMPUTE_NODE` env vars; runs `kubectl label node ... workload=io` and `workload=compute`. Idempotent (`--overwrite`).

**Step 2:** Run on cluster. Verify `kubectl get nodes -l workload=compute`.

**Step 3:** Commit `feat(deploy): node labeling script`.

### Task 4.4: Helm chart skeleton

**Files:**

- Create: `deploy/helm/guitar-detect/Chart.yaml`
- Create: `deploy/helm/guitar-detect/values.yaml` (verbatim from DESIGN.md §7.2)
- Create: `deploy/helm/guitar-detect/values.local.yaml` (overrides for this cluster — node names, host)
- Create: `deploy/helm/guitar-detect/templates/_helpers.tpl`
- Create: `deploy/helm/guitar-detect/templates/namespace.yaml`

**Step 1:** `Chart.yaml`: `apiVersion: v2, name: guitar-detect, version: 0.1.0, appVersion: 0.1.0`.

**Step 2:** `values.yaml` per DESIGN.md §7.2. `_helpers.tpl` with standard `{{ include "guitar-detect.fullname" }}` etc.

**Step 3:** `templates/namespace.yaml`: gated on `.Values.namespace.create` (default true).

**Step 4:** Lint: `helm lint deploy/helm/guitar-detect -f deploy/helm/guitar-detect/values.local.yaml`. Fix until clean.

**Step 5:** Commit `feat(deploy): Helm chart skeleton`.

### Task 4.5: Redis manifest

**Files:**

- Create: `deploy/helm/guitar-detect/templates/redis.yaml`

**Step 1:** StatefulSet (1 replica) + headless Service, Longhorn PVC sized from values, AOF off, `maxmemory 512mb`, `maxmemory-policy allkeys-lru` via configmap. Node selector from values.

**Step 2:** `helm template` → eyeball; `helm install --dry-run` clean.

**Step 3:** Commit `feat(deploy): redis manifest`.

### Task 4.6: Gateway Deployment + Service + Ingress

**Files:**

- Create: `deploy/helm/guitar-detect/templates/gateway-deployment.yaml`
- Create: `deploy/helm/guitar-detect/templates/gateway-service.yaml`
- Create: `deploy/helm/guitar-detect/templates/ingress.yaml`

**Step 1:** Deployment: 1 replica, node selector `workload: io`, resources from values, env from values, image `{{ .Values.image.registry }}/guitar-detect/gateway:{{ .Values.image.tag }}`. Readiness probe `/readyz`, liveness probe `/healthz`, both with reasonable initial delays (worker takes ~20s; gateway is faster — ~5s).

**Step 2:** Service: ClusterIP, port 8000.

**Step 3:** Ingress: Traefik ingressClassName, single host, TLS secret reference. WS path `/ws` and HTTP paths share the same backend (Traefik handles WS automatically).

**Step 4:** `helm install --dry-run` clean; then real install: `helm install guitar-detect deploy/helm/guitar-detect -f deploy/helm/guitar-detect/values.local.yaml --create-namespace -n guitar-detect`.

**Step 5:** Verify pod is Running and `curl -k https://guitars.home.lan/healthz` returns 200.

**Step 6:** Commit `feat(deploy): gateway deployment + ingress`.

### Task 4.7: Inference deployment

**Files:**

- Create: `deploy/helm/guitar-detect/templates/inference-deployment.yaml`

**Step 1:** Deployment: 1 replica (configurable), node selector `workload: compute`, Guaranteed QoS (`requests == limits`), env from values. No service (pure consumer). Readiness probe: a `/healthz` HTTP endpoint on the worker (add a minimal aiohttp/uvicorn endpoint in worker for this — small task; or use a `cat /tmp/ready` exec probe set by the consumer once the model is loaded). Pick exec probe to avoid pulling in FastAPI here.

**Step 2:** Update worker to `touch /tmp/ready` after the pipeline is initialized; readiness probe `exec: cat /tmp/ready`.

**Step 3:** `helm upgrade`. Verify pod runs and lands on the compute node (`kubectl get pods -o wide`).

**Step 4:** Manual gate: load `https://guitars.home.lan` on phone and desktop, full E2E works.

**Step 5:** Commit `feat(deploy): inference deployment + readiness gate`.

### Task 4.8: NetworkPolicies (optional but in scope)

**Files:**

- Create: `deploy/helm/guitar-detect/templates/networkpolicy.yaml`

**Step 1:** Three policies per DESIGN.md §7.3:

- `gateway-ingress`: from Traefik namespace only.
- `redis-ingress`: from gateway and inference pods only.
- `inference-ingress`: deny all.

Gate on `.Values.networkPolicies.enabled` (default true; documented escape hatch).

**Step 2:** Apply, verify still works end-to-end. If broken, inspect Traefik namespace labels — older K3s setups use `kube-system` for ingress controllers.

**Step 3:** Commit `feat(deploy): network policies`.

### Task 4.9: Smoke test script

**Files:**

- Create: `deploy/k3s/smoke-test.sh`

**Step 1:** Script:

- `curl -k https://guitars.home.lan/healthz` expect 200.
- `curl -k https://guitars.home.lan/readyz` expect 200.
- `POST /api/session` then open WS, expect upgrade success (use `websocat`).
- Optional: simulate 30s of frame ingest via the synthetic-producer image and assert detections come back.

**Step 2:** Run; iterate.

**Step 3:** Commit `feat(deploy): smoke test`.

### Task 4.10: Pod kill resilience test

**Step 1:** Manual: open the app on phone, point at a guitar (locked label). On laptop: `kubectl -n guitar-detect delete pod -l app=inference-worker`.

**Step 2:** Observe UI shows "reconnecting" / no detections, then resumes within ~10s once the new worker pod is Ready.

**Step 3:** Document in `docs/E2E_CHECKLIST.md` as test #9.

**Step 4:** Commit `docs: E2E checklist with resilience test`.

### Phase 4 wrap-up

- `git tag phase-4-done`.

---

## Phase 5 — Polish & Gallery

**Phase goal:** Session gallery of unique stable tracks and UX polish per DESIGN.md §8 Phase 5.

**Phase done-when:**

- Pointing camera at multiple guitars in sequence builds a side-panel gallery of thumbnails.
- Clicking a thumbnail highlights that track in the live view (visual emphasis on its bbox).
- HUD fade-in animation, debug panel, color tuning all feel finished.

### Task 5.1: Gallery state (frontend)

**Files:**

- Create: `services/frontend/src/hooks/useGallery.ts`
- Create: `services/frontend/src/hooks/useGallery.test.ts`

**Step 1:** Tests:

- A track that becomes `stable` for the first time triggers gallery capture (capture from `<video>` to a `<canvas>` thumbnail).
- Same `track_id` doesn't capture twice.
- Different track IDs with the same label still both captured (we're showing unique sightings).
- Cleared on `clear()`.

**Step 2-4:** Standard loop. Capture uses an offscreen canvas; thumbnail saved as a `data:image/jpeg` URL (or `Blob` + object URL).

**Step 5:** Commit `feat(frontend): in-memory session gallery`.

### Task 5.2: GalleryPanel component

**Files:**

- Create: `services/frontend/src/components/GalleryPanel.tsx`
- Modify: `services/frontend/src/App.tsx` (lay out video + gallery side-by-side, stacked on narrow screens).

**Step 1:** Renders a scrolling list of thumbnails with label captions. Clicking emits a `highlightTrackId` to App state.

**Step 2:** Update HUD: if `highlightTrackId === t.track_id`, draw a thicker stroke and a brighter glow on that bbox.

**Step 3:** Manual gate: pointing at multiple guitars builds gallery; clicking highlights live track.

**Step 4:** Commit `feat(frontend): gallery panel + highlight interaction`.

### Task 5.3: HUD polish

**Files:**

- Modify: `services/frontend/src/components/HUD.tsx`

**Step 1:** Fade-in animation: bbox opacity 0.3 → 1.0 over age 1→5 (already partially specced in 2.13; verify smooth easing — use `1 - exp(-age/2)` or similar).

**Step 2:** Color tune: verify Gibson gold and Fender white pop on both bright and dark backgrounds; if not, tweak the inner 1px black stroke / shadow.

**Step 3:** Smooth bbox interpolation: between two consecutive frames, lerp the bbox position over `requestAnimationFrame` instead of snapping. Hold last known bbox for ≤ 150ms after a detection drops to avoid flicker.

**Step 4:** Manual eyeball check.

**Step 5:** Commit `polish(frontend): HUD animation, smoothing, color`.

### Task 5.4: Debug panel completion

**Files:**

- Modify: `services/frontend/src/components/DebugPanel.tsx`

**Step 1:** Show: video FPS (rAF rate), detection event FPS, end-to-end latency (now - frame_ts), WebRTC connection state, WS state, recent dropped frames.

**Step 2:** Toggle visible only when `?debug=1`.

**Step 3:** Commit `feat(frontend): debug panel content`.

### Task 5.5: README, DEVELOPMENT.md, DEPLOYMENT.md, TROUBLESHOOTING.md

**Files:**

- Modify: `README.md`
- Create: `docs/DEVELOPMENT.md`
- Create: `docs/DEPLOYMENT.md`
- Create: `docs/TROUBLESHOOTING.md`

**Step 1:** `README.md`: 90-second elevator pitch, screenshot/GIF (placeholder OK), link to docs, quick `make dev` instructions.

**Step 2:** `DEVELOPMENT.md`: local dev loop, hot-reload story, how to add a new guitar model (edit prompts.md), how to run tests.

**Step 3:** `DEPLOYMENT.md`: K3s install, mkcert, registry, Helm install, upgrade, rollback.

**Step 4:** `TROUBLESHOOTING.md`: verbatim from DESIGN.md §10.3 + anything we learned during implementation that wasn't in the design.

**Step 5:** Commit `docs: top-level README + dev/deploy/troubleshooting`.

### Phase 5 wrap-up

- Manual E2E full pass.
- All tests green: `make test` (root target running each service's pytest + frontend vitest).
- `git tag phase-5-done`.
- Tag a release: `git tag v0.1.0`.

---

## Cross-cutting follow-ups (open for later)

Track in `docs/FOLLOWUPS.md`, don't build now:

- Multi-viewer support (SFU)
- Cross-session guitar identity (embedding-based)
- Persistence to a DB
- Fine-tuning on user data
- Prometheus metrics endpoints (DESIGN.md §5.8 v2)
- TURN server if peer connections fail on tricky LANs

---

## Plane / Outline (per user global conventions)

After the plan is approved, create a Plane project (or reuse an existing one) and a Work Item per Phase (5 total), each in `Backlog`. As we start a phase, move to `In Progress`. Capture findings (benchmarks, prompt tweaks) as Work Item comments. Mirror this plan to an Outline document for cross-device reading.

I'll set these up when you give the go-ahead — they need the Plane workspace/project IDs which I'll ask for at that time.
