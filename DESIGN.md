# Guitar Detection System — Design & Implementation Spec

**Target implementer:** Claude Code
**Deployment target:** Home K3s cluster (2 nodes, AMD x86)
**Status:** Ready for implementation

---

## 1. Overview

### 1.1 Purpose
A real-time guitar detection and brand/model classification system. Users connect via a web app from any device with a camera (phone, desktop webcam), and the system identifies guitars in the video feed with a persistent "lock-on" HUD overlay showing brand and model.

### 1.2 Scope
- **Detection target:** Electric guitars (one object class)
- **Classification targets:** 6 specific models across 2 brands
  - Gibson: Les Paul, SG, Explorer, Flying V
  - Fender: Stratocaster, Telecaster
- **Input:** Browser-sourced video via WebRTC (phone or desktop cameras)
- **Output:** Live video with overlay drawn client-side from server-pushed detection events
- **Concurrency:** Single viewer at a time (no multi-viewer multiplexing required)
- **Network scope:** Home LAN only

### 1.3 Non-goals
- Cross-session guitar identity ("this is the same Les Paul as yesterday")
- Detection persistence to a database
- Multi-tenant access / authentication beyond LAN trust
- Mobile native apps (browser is sufficient)
- GPU acceleration (CPU-only by design)

### 1.4 Success criteria
- End-to-end latency from camera capture to overlay draw: **< 150ms p95**
- Inference throughput: **10–15 detection FPS sustained**
- Video display: **24–60 FPS** (limited by source, not pipeline)
- Brand-level classification accuracy: **≥ 85%** on clear, well-lit shots
- Model-level classification accuracy: **≥ 75%** on the 6 target models
- Track persistence: object IDs survive ≥ 2 seconds of occlusion
- Cold start to first detection after pod boot: **< 30 seconds**

---

## 2. Architecture

### 2.1 High-level component diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  Browser (any device on LAN)                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  React SPA                                                   │   │
│  │  • Camera picker (enumerateDevices)                          │   │
│  │  • getUserMedia → <video>                                    │   │
│  │  • WebRTC peer → server                                      │   │
│  │  • WebSocket ← detection events                              │   │
│  │  • <canvas> overlay (HUD render via requestAnimationFrame)   │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ HTTPS / WSS / WebRTC
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│  K3s Cluster                                                        │
│                                                                     │
│  ┌─────────────────────┐         ┌──────────────────────────┐       │
│  │  Traefik Ingress    │         │  Redis (Streams)         │       │
│  │  TLS via mkcert     │         │  • frames:{session_id}   │       │
│  └──────────┬──────────┘         │  • detections:{session_id}│      │
│             │                    └──────────┬───────────────┘       │
│             │  routes to                    │                       │
│             ↓                               │                       │
│  ┌─────────────────────┐    publishes ──────┘                       │
│  │  gateway (FastAPI)  │    frames                                  │
│  │  • Static SPA       │◄──────────┐                                │
│  │  • /api/* endpoints │           │                                │
│  │  • WebSocket /ws    │           │                                │
│  │  • WebRTC signaling │           │                                │
│  │  • aiortc peer      │  publishes detections                      │
│  │  Node: io           │           │                                │
│  └──────────┬──────────┘           │                                │
│             │                      │                                │
│             │ session frames       │                                │
│             ↓                      │                                │
│  ┌──────────────────────────────────────────┐                       │
│  │  inference-worker (replicas)             │                       │
│  │  • Reads frames:* (consumer group)       │                       │
│  │  • YOLOv8n-oiv7 (guitar detector)        │                       │
│  │  • ByteTrack (persistent IDs)            │                       │
│  │  • MobileCLIP (zero-shot classifier)     │                       │
│  │  • Rolling vote per track                │                       │
│  │  • Publishes detections:{session_id}     │                       │
│  │  Node: compute                           │                       │
│  └──────────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component responsibilities

| Component | Responsibility | Node | Replicas |
|-----------|---------------|------|----------|
| `gateway` | Serves SPA, WebRTC signaling, WebRTC peer (aiortc), frame ingest, WebSocket out | `io` (mobile AMD) | 1 |
| `inference-worker` | YOLO detection, ByteTrack, MobileCLIP classification, vote smoothing | `compute` (Ryzen 7) | 1–2 |
| `redis` | Frame and detection message bus (Streams) | `io` | 1 |
| `traefik` | TLS termination, routing | (K3s default) | (K3s default) |

### 2.3 Why this split

- **Gateway co-locates signaling + WebRTC ingest** to avoid an extra hop. aiortc decoding is light (~5ms/frame); pairing it with signaling keeps the path simple.
- **Inference is isolated** so it can be scaled, restarted, or upgraded without touching the user-facing session.
- **Redis Streams** provides backpressure (consumer groups, MAXLEN trim) and natural decoupling. Frames are ephemeral (TTL ~2s); we never need durability.
- **Node pinning** uses the Ryzen 7's cores fully for inference and keeps the mobile chip on light I/O work.

### 2.4 Data flow per frame

1. Browser captures camera frame via WebRTC track.
2. aiortc in `gateway` decodes the frame to a numpy array (BGR, HxWx3).
3. `gateway` JPEG-encodes (quality 75) and `XADD`s to `frames:{session_id}` Redis stream with `MAXLEN ~ 30`.
4. `inference-worker` consumer (in group `inference`) reads the frame.
5. Worker runs YOLO → ByteTrack → (for new/unstable tracks) MobileCLIP.
6. Worker updates internal rolling vote state and `XADD`s a detection event to `detections:{session_id}`.
7. `gateway` (subscribed to `detections:*` for active sessions) forwards events over the session's WebSocket.
8. Browser receives JSON, updates canvas overlay in next animation frame.

### 2.5 Session lifecycle

- Browser opens page → JS generates a UUIDv4 `session_id`, stored in memory only.
- Browser POSTs `/api/session` with `{session_id}` to register; gateway creates Redis streams.
- Browser opens WebSocket `/ws?session_id=...` for detection events.
- Browser initiates WebRTC offer at `/api/webrtc/offer` with `{session_id, sdp}`.
- Gateway establishes peer connection, starts forwarding frames to Redis.
- On WebSocket close OR no frames for 10s: gateway tears down peer, deletes Redis streams.

---

## 3. Technology Choices

### 3.1 Inference stack

| Concern | Choice | Rationale |
|---------|--------|-----------|
| Object detection | YOLOv8n (Ultralytics) | Lightweight, fast on CPU, mature tooling |
| Detection weights | `yolov8n-oiv7.pt` (Open Images V7) | Has a "Guitar" class out of the box — zero training needed |
| Tracking | ByteTrack (built into Ultralytics `model.track`) | Robust, free of charge, no extra dependency |
| Classification | OpenCLIP MobileCLIP-S0 | Zero-shot via text prompts, INT8-quantizable, ~5–8ms/crop |
| Runtime | OpenVINO 2024.4+ | Best CPU performance on Intel/AMD x86 |
| Quantization | INT8 post-training (OpenVINO) | ~2–3× speedup, minimal accuracy loss for our use case |

### 3.2 Web stack

| Concern | Choice | Rationale |
|---------|--------|-----------|
| Backend framework | FastAPI | Native async, WebSocket support, simple |
| WebRTC server | aiortc | Mature Python WebRTC, integrates with asyncio |
| Frontend framework | React 18 + Vite | Fast dev loop, small bundle, ergonomic |
| Frontend language | TypeScript | Catch overlay/event shape bugs early |
| Styling | Tailwind CSS | Quick HUD styling without CSS files proliferating |
| State | React hooks only (no Redux) | App is small enough |

### 3.3 Infrastructure

| Concern | Choice | Rationale |
|---------|--------|-----------|
| Orchestration | K3s | User constraint |
| Storage | Longhorn (RWX where needed) | User constraint |
| Bus | Redis 7 (Streams) | Simple, fast, low ops burden |
| Ingress | Traefik (K3s default) | Already installed |
| TLS | mkcert + manual cert installation | Required for `getUserMedia` on phones; LAN-only |
| Image registry | Local `registry:2` deployment | Avoid Docker Hub rate limits |
| Packaging | Helm chart | Easy iteration, value-driven config |

---

## 4. Repository Structure

```
guitar-detect/
├── README.md
├── DESIGN.md                          # this document
├── Makefile                           # common dev tasks
├── docker-compose.yml                 # local-cluster equivalent for dev
├── .env.example
│
├── services/
│   ├── gateway/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py                # FastAPI entry
│   │   │   ├── config.py              # pydantic-settings
│   │   │   ├── session.py             # session lifecycle mgmt
│   │   │   ├── webrtc.py              # aiortc peer mgmt
│   │   │   ├── redis_io.py            # Redis Streams helpers
│   │   │   ├── websocket.py           # detection event forwarding
│   │   │   └── static/                # built SPA copied here at image build
│   │   └── tests/
│   │       ├── test_session.py
│   │       ├── test_redis_io.py
│   │       └── test_api.py
│   │
│   ├── inference-worker/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py                # worker entry / consumer loop
│   │   │   ├── config.py
│   │   │   ├── detector.py            # YOLO + tracker wrapper
│   │   │   ├── classifier.py          # MobileCLIP wrapper
│   │   │   ├── voting.py              # rolling vote per track
│   │   │   ├── pipeline.py            # orchestrates detect→track→classify→vote
│   │   │   └── models/                # downloaded/quantized weights (gitignored)
│   │   ├── scripts/
│   │   │   ├── download_models.py     # fetch + export to OpenVINO INT8
│   │   │   └── benchmark.py           # measure per-stage latency
│   │   └── tests/
│   │       ├── test_voting.py
│   │       ├── test_classifier.py
│   │       └── test_pipeline.py
│   │
│   └── frontend/
│       ├── Dockerfile                 # multi-stage build → static assets
│       ├── package.json
│       ├── vite.config.ts
│       ├── tsconfig.json
│       ├── tailwind.config.js
│       ├── index.html
│       ├── src/
│       │   ├── main.tsx
│       │   ├── App.tsx
│       │   ├── api/
│       │   │   ├── session.ts
│       │   │   └── webrtc.ts
│       │   ├── components/
│       │   │   ├── CameraPicker.tsx
│       │   │   ├── VideoStage.tsx     # <video> + <canvas> overlay
│       │   │   ├── HUD.tsx            # detection box rendering
│       │   │   └── DebugPanel.tsx     # FPS, latency, connection state
│       │   ├── hooks/
│       │   │   ├── useWebRTC.ts
│       │   │   ├── useDetections.ts   # WebSocket subscription
│       │   │   └── useCamera.ts
│       │   ├── types/
│       │   │   └── detection.ts
│       │   └── styles/
│       │       └── index.css
│       └── tests/
│           └── overlay.test.tsx
│
├── deploy/
│   ├── helm/
│   │   └── guitar-detect/
│   │       ├── Chart.yaml
│   │       ├── values.yaml
│   │       ├── values.local.yaml      # overrides for home cluster
│   │       └── templates/
│   │           ├── _helpers.tpl
│   │           ├── namespace.yaml
│   │           ├── redis.yaml
│   │           ├── gateway-deployment.yaml
│   │           ├── gateway-service.yaml
│   │           ├── inference-deployment.yaml
│   │           ├── pvc-models.yaml
│   │           ├── configmap.yaml
│   │           ├── secrets.yaml       # tls secret reference
│   │           ├── ingress.yaml
│   │           └── networkpolicy.yaml
│   └── k3s/
│       ├── README.md                  # node labeling, mkcert, registry setup
│       ├── label-nodes.sh
│       ├── install-registry.sh
│       └── install-mkcert-cert.sh
│
└── docs/
    ├── DEVELOPMENT.md                 # how to run locally
    ├── DEPLOYMENT.md                  # how to deploy to K3s
    ├── TROUBLESHOOTING.md
    └── prompts.md                     # the CLIP prompt list, versioned
```

---

## 5. Detailed Specifications

### 5.1 Type definitions (shared semantics)

The same logical types appear in Python (gateway, worker) and TypeScript (frontend). Keep field names identical.

#### `DetectionEvent` (worker → gateway → browser)
```typescript
type DetectionEvent = {
  session_id: string;
  frame_id: number;            // monotonic per session, set by gateway at ingest
  frame_ts: number;            // unix ms, set by gateway at ingest
  inference_ts: number;        // unix ms, set by worker on emit
  tracks: TrackDetection[];
};

type TrackDetection = {
  track_id: number;            // assigned by ByteTrack, stable per object
  bbox: [number, number, number, number];  // [x1, y1, x2, y2] normalized 0..1
  detection_confidence: number;            // YOLO confidence 0..1
  label: ClassificationLabel | null;       // null while vote is still warming up
  stable: boolean;                         // true once vote window is full
  age_frames: number;                      // frames since track first appeared
};

type ClassificationLabel = {
  brand: "Gibson" | "Fender" | "Unknown";
  model: "Les Paul" | "SG" | "Explorer" | "Flying V" | "Stratocaster" | "Telecaster" | "Unknown";
  confidence: number;                      // smoothed vote score 0..1
};
```

#### `FrameMessage` (gateway → Redis → worker)
Stored as Redis Stream fields:
```
session_id: str
frame_id: int (stringified)
frame_ts: int (unix ms, stringified)
width: int (stringified)
height: int (stringified)
jpeg: bytes      # JPEG-encoded BGR frame, quality 75
```

### 5.2 HTTP / WebSocket API (gateway)

Base path: `/api`

#### `POST /api/session`
Register a new session.
- Request: `{ "session_id": "uuid-v4" }`
- Response: `200 { "ok": true }` or `409` if already exists
- Side effect: creates Redis streams with TTL; allocates session state

#### `DELETE /api/session/{session_id}`
Tear down a session (also happens automatically on WS close or 10s idle).
- Response: `204`

#### `POST /api/webrtc/offer`
Exchange WebRTC SDP.
- Request: `{ "session_id": str, "sdp": str, "type": "offer" }`
- Response: `{ "sdp": str, "type": "answer" }`
- Side effect: creates aiortc peer, registers `on_track` handler

#### `WS /ws?session_id={session_id}`
Server pushes `DetectionEvent` JSON messages. Client-to-server messages are ignored except for `{"type": "ping"}` (server replies `{"type": "pong"}`).

#### `GET /healthz` and `GET /readyz`
Standard k8s health endpoints.
- `/healthz`: returns 200 if process is alive
- `/readyz`: returns 200 only when Redis is reachable

### 5.3 Redis schema

| Key | Type | TTL | Purpose |
|-----|------|-----|---------|
| `frames:{session_id}` | Stream, MAXLEN ~ 30 | (implicit via MAXLEN) | Frame bus, gateway → worker |
| `detections:{session_id}` | Stream, MAXLEN ~ 100 | (implicit via MAXLEN) | Detection events, worker → gateway |
| `session:{session_id}` | Hash | 60s sliding | Session metadata (created_ts, last_frame_ts) |
| `sessions:active` | Set | — | Active session IDs (for worker discovery) |

Consumer groups:
- Group `inference` on `frames:*` — workers join this group, read with `XREADGROUP`
- Gateway reads `detections:*` with `XREAD` (no group needed; only one gateway)

Worker discovers sessions: on a 1-second tick, `SMEMBERS sessions:active` → ensure a consumer reading each session's stream. Drop consumers for missing sessions.

### 5.4 Inference pipeline detail

#### Detector (`detector.py`)
- Load OpenVINO-exported `yolov8n-oiv7-int8`
- Inference at **416×416** (resize from incoming frame)
- Filter results to class "Guitar" (OIv7 class ID — confirm at load time, fail fast if missing)
- Confidence threshold: **0.35** (tunable via env `DETECT_CONF`)
- IoU NMS threshold: **0.5**
- Wrap with Ultralytics `model.track(persist=True, tracker="bytetrack.yaml")` for ID assignment
- Returns: list of `(track_id, bbox_xyxy_pixels, det_conf)` per frame

#### Classifier (`classifier.py`)
- Load OpenVINO-exported MobileCLIP-S0 (image and text towers, both INT8 where possible — text tower may stay FP16)
- **Text features precomputed once at startup** from `docs/prompts.md`
- Per inference call:
  - Crop bbox from original-resolution frame (pre-resize)
  - Pad to square, resize to 224×224 (MobileCLIP-S0 input size)
  - Encode with image tower
  - Cosine similarity vs precomputed text features
  - Softmax with temperature 100 (CLIP standard) → label probabilities
  - Map prompt index → (brand, model)
- Returns: list of (brand, model, confidence) candidates sorted by confidence; top-1 used

**Prompt file** (`docs/prompts.md`) — versioned, editable without code change:
```yaml
prompts:
  - text: "a photograph of a Gibson Les Paul electric guitar"
    brand: Gibson
    model: Les Paul
  - text: "a photograph of a Gibson SG electric guitar"
    brand: Gibson
    model: SG
  - text: "a photograph of a Gibson Explorer electric guitar"
    brand: Gibson
    model: Explorer
  - text: "a photograph of a Gibson Flying V electric guitar"
    brand: Gibson
    model: Flying V
  - text: "a photograph of a Fender Stratocaster electric guitar"
    brand: Fender
    model: Stratocaster
  - text: "a photograph of a Fender Telecaster electric guitar"
    brand: Fender
    model: Telecaster
  - text: "a photograph of an acoustic guitar"
    brand: Unknown
    model: Unknown
  - text: "a photograph of a bass guitar"
    brand: Unknown
    model: Unknown
  - text: "a photograph of a different electric guitar"
    brand: Unknown
    model: Unknown
```

The "Unknown" prompts act as a rejection class — if their combined score wins, we emit `Unknown` rather than forcing one of the 6.

#### Voting (`voting.py`)
Per track ID, maintain a `deque(maxlen=15)` of recent classifications. On each new sample:
- Append `(brand, model, confidence)` tuple
- Compute weighted vote: each entry contributes its confidence to its (brand, model) bucket
- Winning bucket = highest total weight
- Smoothed confidence = winner_weight / sum_all_weights
- **Stable flag** = `len(deque) >= 8 AND smoothed_confidence >= 0.55`
- If `Unknown` wins → emit `label = null` with `stable = false` (don't lock onto wrong ID)

Tracks pruned from memory if not seen for 90 frames.

#### When to classify (cost optimization)
- **Always** classify on the first 5 frames of a new track (warm up the vote)
- **Every 6th frame** thereafter for unstable tracks
- **Every 30th frame** for stable tracks (drift check)
- Skip if bbox area < 0.5% of frame (too small to classify reliably)

#### Pipeline loop (`pipeline.py`)
```
loop forever:
  msgs = XREADGROUP from frames:* (block 100ms)
  for each msg:
    frame = decode_jpeg(msg.jpeg)
    detections = detector.detect_and_track(frame)
    out_tracks = []
    for det in detections:
      should_classify = decide_classify(det, vote_state[det.track_id])
      if should_classify:
        crop = crop_bbox(frame, det.bbox)
        label_raw = classifier.classify(crop)
        vote_state[det.track_id].update(label_raw)
      smoothed = vote_state[det.track_id].current()
      out_tracks.append(make_track_detection(det, smoothed))
    publish_detection_event(session_id, out_tracks)
    XACK message
  prune_old_tracks()
```

### 5.5 Gateway detail

#### WebRTC frame ingest (`webrtc.py`)
- One `RTCPeerConnection` per session
- On `track` event of kind `"video"`:
  - Spawn an asyncio task that `recv()`s frames in a loop
  - Convert `av.VideoFrame` → numpy BGR
  - **Frame rate limiter:** drop frames if last publish was < 33ms ago (cap ingest at ~30 FPS regardless of camera FPS, since inference can't keep up beyond that anyway)
  - JPEG encode (cv2.imencode quality 75)
  - `XADD frames:{session_id} MAXLEN ~ 30 ...`
- On peer connection state `failed` / `closed`: trigger session teardown

#### Detection forwarder (`websocket.py`)
- One asyncio task per active WebSocket
- Loops `XREAD detections:{session_id}` with block 100ms
- Sends each event as JSON over the WebSocket
- Exits when WS closes

### 5.6 Frontend detail

#### Page flow
1. Landing screen: heading + camera picker dropdown (populated from `enumerateDevices()`)
2. User selects camera → "Start" button enables
3. On Start:
   - Request camera permission via `getUserMedia({ video: { deviceId } })`
   - Generate `session_id` (`crypto.randomUUID()`)
   - POST `/api/session`
   - Open WS to `/ws?session_id=...`
   - Create `RTCPeerConnection`, add camera track, create offer, POST to `/api/webrtc/offer`, set remote answer
   - Render `<video>` (autoplay, muted) + `<canvas>` overlay
4. Detection events flow in via WS → update React state → canvas redraws via `requestAnimationFrame`

#### Canvas overlay rendering (`HUD.tsx`)
For each track:
- Box stroke color: brand → color map
  - Gibson: `#C8A45C` (gold)
  - Fender: `#F5F5F5` (white)
  - Unknown / null: `#888888` (gray)
- Stroke width: 3px, with a 1px black inner stroke for contrast on any background
- Box opacity ramps from 0.3 → 1.0 over first 5 frames of track lifetime
- Label rendered above box (or below if near top edge):
  - Font: bold 14px system font
  - Background: black with 70% opacity, 4px padding
  - Text: `[#{track_id}] {brand} {model} · {confidence|%}`
  - If `!stable`: italic text reading `Analyzing…` instead of label
- Optional debug overlay (toggled via `?debug=1`): FPS, latency (frame_ts → now), connection state

#### Coordinate handling
Server emits normalized bbox (0..1). Browser multiplies by current `<video>` `clientWidth`/`clientHeight` (after `object-fit: contain` letterboxing math). Provide a utility `denormalizeBbox(bbox, videoRect)` and unit-test it.

### 5.7 Configuration (env vars)

#### Gateway
| Var | Default | Description |
|-----|---------|-------------|
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection |
| `MAX_INGEST_FPS` | `30` | Frame rate cap before publishing |
| `JPEG_QUALITY` | `75` | Encoding quality |
| `SESSION_IDLE_TIMEOUT_S` | `10` | Tear down if no frames received |
| `LOG_LEVEL` | `INFO` | |

#### Worker
| Var | Default | Description |
|-----|---------|-------------|
| `REDIS_URL` | `redis://redis:6379/0` | |
| `DETECT_CONF` | `0.35` | YOLO confidence threshold |
| `DETECT_IOU` | `0.5` | NMS IoU |
| `DETECT_IMGSZ` | `416` | YOLO input size |
| `CLIP_INPUT_SIZE` | `224` | MobileCLIP input size |
| `VOTE_WINDOW` | `15` | Rolling vote deque length |
| `VOTE_STABLE_MIN` | `8` | Min entries before `stable=true` |
| `VOTE_STABLE_CONF` | `0.55` | Min confidence for `stable=true` |
| `MODELS_DIR` | `/models` | Path to OpenVINO model files |
| `PROMPTS_FILE` | `/config/prompts.yaml` | CLIP prompts |
| `OPENVINO_DEVICE` | `CPU` | OV inference device |
| `OPENVINO_THREADS` | `0` (auto) | Threads per inference |

### 5.8 Logging and observability
- Structured JSON logs (loguru or stdlib logging with `python-json-logger`)
- Each log line includes `session_id` when in a session context
- Worker logs per-stage timing every 60 seconds: detect_ms, classify_ms, total_ms (p50/p95)
- Prometheus metrics (optional v2, but expose endpoints): `frames_ingested_total`, `frames_dropped_total`, `inference_duration_seconds`, `active_sessions`

---

## 6. Container Specifications

### 6.1 Gateway image

Base: `python:3.11-slim` (final stage)
Build stage: install system deps (`libavcodec`, `libavformat`, `libavutil`, `libswscale`, `libsrtp2-dev`, `libopus0`, `libvpx7`) for aiortc.
Frontend assets are built in a separate Node stage and copied into `app/static/`.
Multi-stage; final size target < 350MB.

Entrypoint: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1`

### 6.2 Inference worker image

Base: `openvino/ubuntu22_runtime:2024.4.0`
Add: `python3-pip`, then `pip install` of `ultralytics`, `open-clip-torch`, `numpy`, `opencv-python-headless`, `redis`, `pyyaml`, `loguru`.

Models baked in at build time via `scripts/download_models.py` (saves PyTorch → exports to OpenVINO INT8 → writes to `/models/`). This avoids first-boot model download in cluster.

Final size target < 1.5GB (OpenVINO base alone is ~800MB).

Entrypoint: `python -m app.main`

### 6.3 Frontend image

Not actually a runtime image — built as a Node multi-stage and the `dist/` output is **copied into the gateway image** at build time. The gateway serves it from `app/static/`.

---

## 7. Kubernetes / Helm Specification

### 7.1 Node labeling (one-time setup)

```bash
kubectl label node <mobile-node-name>  workload=io
kubectl label node <ryzen-node-name>   workload=compute
```

### 7.2 Helm chart values (defaults)

```yaml
# values.yaml
namespace: guitar-detect

image:
  registry: registry.local:5000      # local K3s registry
  pullPolicy: IfNotPresent
  tag: "0.1.0"

gateway:
  image: guitar-detect/gateway
  replicas: 1
  nodeSelector:
    workload: io
  resources:
    requests:
      cpu: "500m"
      memory: "512Mi"
    limits:
      cpu: "2"
      memory: "1Gi"
  env:
    LOG_LEVEL: INFO
    MAX_INGEST_FPS: "30"

inference:
  image: guitar-detect/inference-worker
  replicas: 1
  nodeSelector:
    workload: compute
  resources:
    requests:
      cpu: "3"
      memory: "2Gi"
    limits:
      cpu: "6"
      memory: "4Gi"
  env:
    DETECT_CONF: "0.35"
    DETECT_IMGSZ: "416"
    VOTE_WINDOW: "15"

redis:
  enabled: true
  image: redis:7-alpine
  nodeSelector:
    workload: io
  storage:
    size: 1Gi
    storageClass: longhorn

ingress:
  enabled: true
  className: traefik
  host: guitars.home.lan
  tls:
    enabled: true
    secretName: guitars-tls

models:
  pvc:
    enabled: false    # default: baked into image
    size: 2Gi
    storageClass: longhorn
```

### 7.3 Critical manifest details

**Gateway Service** must be `ClusterIP` (default). All traffic via Traefik ingress (TLS terminated there). WebSocket support is automatic with Traefik 2.x — no extra annotation needed for K3s default.

**Ingress** must include both HTTP and WS routes (same `host`, different paths handled by FastAPI internally).

**Inference Deployment** uses Guaranteed QoS class (`requests == limits` for CPU and memory) to avoid throttling jitter.

**Redis** uses a single replica with a Longhorn PVC. AOF disabled (data is ephemeral). `maxmemory-policy allkeys-lru` and `maxmemory 512mb` to bound memory.

**Network policies** (optional but recommended):
- `gateway` ingress: from `traefik` namespace only
- `redis` ingress: from `gateway` and `inference` pods only
- `inference` ingress: none (no inbound traffic)

### 7.4 Cluster prerequisites

Document in `deploy/k3s/README.md`:
1. K3s version: 1.28+ recommended
2. Longhorn installed and `longhorn` storage class default
3. Local registry running: `deploy/k3s/install-registry.sh` deploys `registry:2` at `registry.local:5000`, K3s configured with `/etc/rancher/k3s/registries.yaml` pointing to it
4. mkcert: generate cert for `guitars.home.lan`, create k8s secret `guitars-tls`, install root CA on each viewing device
5. DNS: add `guitars.home.lan` → K3s ingress IP in router or `/etc/hosts` on each device

---

## 8. Implementation Plan (Phased)

Each phase is independently testable and ends in a working artifact.

### Phase 1 — Inference core (worker only, no K8s, no web)
**Goal:** Validate the YOLO + ByteTrack + MobileCLIP + voting pipeline on local webcam.
**Deliverables:**
- `services/inference-worker/` with `pipeline.py` runnable as a script
- Test harness: opens local webcam via OpenCV, runs full pipeline, draws boxes on a window
- `scripts/download_models.py` produces OpenVINO INT8 weights
- Latency benchmark in `scripts/benchmark.py`

**Done when:**
- Webcam shows guitars highlighted with brand/model labels
- p50 detect+classify+track latency < 50ms on Ryzen 7
- Track IDs remain stable across short occlusions

### Phase 2 — Gateway + frontend (single-node dev, docker-compose)
**Goal:** End-to-end browser → server → browser overlay.
**Deliverables:**
- `services/gateway/` with full API and WebRTC ingest
- `services/frontend/` SPA with camera picker and overlay
- `docker-compose.yml` running gateway + worker + redis
- All run on developer laptop, viewed at `https://localhost:8000` (mkcert dev cert)

**Done when:**
- Open browser, pick camera, point at guitar, see HUD overlay
- E2E latency < 200ms p95 on dev machine

### Phase 3 — Containerization & local cluster validation
**Goal:** Production-shaped images, ready to deploy.
**Deliverables:**
- Final Dockerfiles for both services
- Frontend baked into gateway image
- `docker-compose.yml` updated to use built images (no live mounts)
- Image size budgets met (gateway <350MB, worker <1.5GB)

**Done when:**
- `docker compose up` from clean state works
- Health checks pass
- Same E2E latency as Phase 2

### Phase 4 — K3s deployment
**Goal:** Running on the home cluster.
**Deliverables:**
- Local registry deployed, push scripts in Makefile
- Helm chart deployable with `helm install`
- `deploy/k3s/README.md` covers node labeling, mkcert, DNS setup
- Smoke test script that hits `/healthz`, `/readyz`, opens a WS

**Done when:**
- `https://guitars.home.lan` works from phone and desktop on LAN
- Camera permission prompts and overlay works on both
- Pods placed on expected nodes (`kubectl get pods -o wide` confirms)
- Restart of inference pod doesn't break active session (session reconnects within 5s)

### Phase 5 — Polish & gallery
**Goal:** "Gallery of guitars seen this session" + UX refinement.
**Deliverables:**
- In-memory gallery: thumbnail crops of each unique track that became stable
- Side panel in UI showing gallery, click to highlight that track in the live view
- Cleared on session end (no persistence)
- HUD polish: fade-in animation, color tuning, debug panel

**Done when:**
- Pointing camera at multiple guitars in sequence builds the gallery
- Gallery thumbnails are recognizable
- UI feels finished

---

## 9. Testing Strategy

### 9.1 Test pyramid

```
       /\
      /E2E\           <- Phase-gating manual tests (browser, real camera)
     /------\
    / Integ. \        <- docker-compose: gateway ↔ worker ↔ redis
   /----------\
  /   Unit     \      <- Pure-function tests, the bulk of automated coverage
 /--------------\
```

### 9.2 Unit tests (per service, run on every commit)

**Inference worker:**
- `test_voting.py`: vote with handcrafted inputs → expected stable label
  - Single brand consistently winning → stable in 8 frames
  - Flapping between two labels → not stable
  - Unknown winning → stable but `label=null`
  - Pruning of old entries
- `test_classifier.py`: feed known guitar images (committed to repo as test fixtures, ~10 images), assert top-1 label matches expected brand/model
  - Tolerance: 80% of fixtures must classify correctly (CLIP is fuzzy)
- `test_detector.py`: mock OpenVINO output, verify bbox parsing, NMS, class filtering
- `test_pipeline.py`: integration of detector mock + real classifier + real voting on fixture images

**Gateway:**
- `test_session.py`: lifecycle — create, idle timeout, explicit delete, double-create rejection
- `test_redis_io.py`: stream publish/consume round-trip against a fakeredis or real ephemeral redis
- `test_api.py`: FastAPI TestClient — endpoint contracts, validation errors, health checks
- `test_webrtc.py`: SDP offer/answer round-trip with a mock peer (skip if aiortc test harness too heavy)

**Frontend:**
- `overlay.test.tsx`: render `HUD` with fixed `tracks` props, assert canvas commands via mocked context
- `denormalizeBbox.test.ts`: math for object-fit letterboxing
- `useDetections.test.ts`: WebSocket reconnect logic

### 9.3 Integration tests (run before commits to main)

`docker-compose -f docker-compose.test.yml up` runs:
- Redis
- Gateway
- Worker
- A **synthetic frame producer** container that POSTs to `/api/session`, opens a WS, then publishes JPEG fixtures directly to Redis (bypassing WebRTC for determinism) at 15 FPS
- An **assertion container** that subscribes to the WS and validates:
  - Detection events arrive
  - Track IDs are stable across consecutive frames of the same fixture
  - Stable labels emerge within 1 second of feeding a clear guitar fixture
  - Session teardown cleans up Redis streams

### 9.4 End-to-end manual test plan (gates each phase)

A scripted checklist in `docs/E2E_CHECKLIST.md`:

| # | Test | Pass criteria |
|---|------|---------------|
| 1 | Open landing page on desktop Chrome | Camera picker populated, mkcert green padlock |
| 2 | Same on iOS Safari | Same behavior (mkcert CA installed) |
| 3 | Point at Stratocaster (provide reference image) | Stable `Fender Stratocaster` within 2s |
| 4 | Point at Les Paul | Stable `Gibson Les Paul` within 2s |
| 5 | Point at acoustic | `Analyzing…` then stays unlabeled (Unknown wins) |
| 6 | Walk guitar out of frame, back in | Same track ID if <2s, new ID if >2s |
| 7 | Two guitars in frame | Two independent track IDs and labels |
| 8 | Network blip (toggle wifi) | UI shows "reconnecting", recovers within 5s |
| 9 | Kill inference pod mid-session | UI continues, detections resume within ~10s |
| 10 | Idle session (close tab) | Server logs show session teardown within 10s |

### 9.5 Performance tests

`services/inference-worker/scripts/benchmark.py`:
- Feeds a 60-second loop of fixture frames at 30 FPS
- Reports: detect_ms p50/p95, classify_ms p50/p95, end_to_end_ms p50/p95, dropped_frames
- Pass criteria documented as "phase done" definitions above

---

## 10. Operations Runbook

### 10.1 Initial deployment

```bash
# 1. Prepare cluster (one-time)
./deploy/k3s/label-nodes.sh
./deploy/k3s/install-registry.sh
./deploy/k3s/install-mkcert-cert.sh    # generates cert, creates secret

# 2. Build & push images
make build-images
make push-images

# 3. Install
helm install guitar-detect ./deploy/helm/guitar-detect \
  -f ./deploy/helm/guitar-detect/values.local.yaml \
  --create-namespace \
  --namespace guitar-detect

# 4. Verify
kubectl -n guitar-detect get pods -o wide
kubectl -n guitar-detect logs -l app=gateway --tail=50
curl -k https://guitars.home.lan/healthz
```

### 10.2 Upgrade

```bash
make build-images TAG=0.2.0
make push-images TAG=0.2.0
helm upgrade guitar-detect ./deploy/helm/guitar-detect \
  -f ./deploy/helm/guitar-detect/values.local.yaml \
  --set image.tag=0.2.0
```

### 10.3 Common issues

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Browser: no camera in picker | Not on HTTPS or mkcert CA missing on device | Install mkcert root CA on the device |
| Browser: WebRTC connection fails | Traefik not forwarding WS / UDP issues | Check Traefik logs; confirm UDP not needed (aiortc TCP fallback) |
| No detections appearing | Worker can't reach Redis OR no frames flowing | `kubectl logs` worker; check `XLEN frames:*` in redis |
| Wrong classifications | CLIP prompts off OR detector cropping wrong | Tweak `prompts.yaml`; verify bbox padding in `classifier.py` |
| Worker pod OOMKilled | Model memory grew (multi-replica on same node?) | Increase limit OR reduce replicas |
| High latency | Inference falling behind | Check `frames_dropped_total`; reduce `MAX_INGEST_FPS` or `DETECT_IMGSZ` |
| Pod scheduled on wrong node | Node labels missing or mismatched | Re-run `label-nodes.sh`, `kubectl describe node` |

### 10.4 Debugging commands

```bash
# Watch live detection events for a session
kubectl -n guitar-detect exec -it deploy/redis -- \
  redis-cli XREAD BLOCK 0 STREAMS detections:{session_id} '$'

# Check frame backlog
kubectl -n guitar-detect exec -it deploy/redis -- \
  redis-cli XLEN frames:{session_id}

# Stream worker logs
kubectl -n guitar-detect logs -l app=inference-worker -f --tail=100

# Force session cleanup
kubectl -n guitar-detect exec -it deploy/gateway -- \
  curl -X DELETE http://localhost:8000/api/session/{session_id}
```

### 10.5 Tuning checklist

If accuracy is poor:
1. Inspect fixture-based unit tests — is the classifier itself wrong, or is detection cropping badly?
2. Tweak `prompts.yaml` — try variants (e.g., "a black Gibson Les Paul" vs generic)
3. Lower `DETECT_CONF` to surface more candidate guitars
4. Increase `VOTE_WINDOW` for more smoothing (at cost of slower lock-on)

If latency is poor:
1. `benchmark.py` to localize the slow stage
2. Drop `DETECT_IMGSZ` from 416 → 320 (significant speedup)
3. Reduce `MAX_INGEST_FPS` to 20 (frees CPU; overlay still feels live)
4. Confirm INT8 quantization actually applied (`benchmark.py` logs model precision)

---

## 11. Open Questions / Future Work

These are explicitly **out of scope for v1** but noted so Claude Code doesn't accidentally over-build:

- Multi-viewer support (would need SFU pattern: mediasoup / Janus)
- Cross-session persistence ("recognize this Les Paul tomorrow") — needs embedding-based instance matching, not just class labels
- Detection event persistence to a DB
- Fine-tuning the classifier on user-supplied data
- Mobile-native app
- Authentication beyond LAN trust
- Multi-GPU or actual GPU support (architecture currently CPU-only)
- Beyond 6 models — adding classes is just editing `prompts.yaml`, but accuracy degrades with more visually-similar classes

---

## 12. Glossary

| Term | Meaning |
|------|---------|
| **Track** | A persistent identity assigned by ByteTrack to a detected object across frames |
| **Stable** | A track whose classification has converged (see voting spec §5.4) |
| **Session** | One browser tab's connection: WebRTC peer + WebSocket + Redis streams |
| **OIv7** | Open Images Dataset V7 — provides "Guitar" class for YOLO pretraining |
| **OpenVINO** | Intel's CPU-optimized inference runtime (works on AMD x86 too) |
| **mkcert** | Tool for generating locally-trusted dev certificates |
| **Longhorn** | Kubernetes-native distributed block storage (user's chosen storage class) |
