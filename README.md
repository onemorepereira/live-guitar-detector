# Guitar Detection System

Real-time, browser-based guitar detection and brand/model classification.
Users connect a camera (phone or desktop) via WebRTC; a FastAPI gateway streams
frames to a CPU-only inference worker that runs YOLOv8n + MobileCLIP zero-shot
classification with per-track rolling votes. Detections flow back over a
WebSocket and render as a "lock-on" HUD overlay on the live video.
Deploys to a 2-node K3s home cluster (`io` + `compute`).

## Documentation

- Design and implementation spec: [`DESIGN.md`](./DESIGN.md)
- Implementation plans: [`docs/plans/`](./docs/plans/)
