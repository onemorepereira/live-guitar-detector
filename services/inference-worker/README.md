# Inference Worker

YOLO-based guitar detection plus MobileCLIP zero-shot brand/model
classification. Phase 1 ships a local webcam demo as the manual
acceptance gate (DESIGN.md §8 Phase 1); Phase 2 will replace the demo
loop with the Redis Streams consumer.

## One-time setup

```bash
cd services/inference-worker
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,export]"
python scripts/download_models.py all --out app/models/
```

The `download_models.py` step fetches YOLOv8n-OIv7 weights and the
MobileCLIP-S1 checkpoint, then exports both to OpenVINO IR under
`app/models/`. See `scripts/README.md` for sub-commands.

## Webcam demo (Phase 1 manual gate)

```bash
cd services/inference-worker
source .venv/bin/activate
python -m app.main --webcam 0
```

An OpenCV window opens; point a webcam at a guitar (a printed photo is
fine for ad-hoc testing). Within ~2 s a brand/model label should "lock
on" with brand-coloured bounding box:

| Brand   | Box colour             |
| ------- | ---------------------- |
| Gibson  | `#C8A45C` warm gold    |
| Fender  | `#F5F5F5` off-white    |
| Unknown | `#888888` neutral grey |

Press `q` (with the OpenCV window focused) to quit. The console prints
`frames/s`, `avg`, and `p95` latency once per second so you can sanity
check performance against the Phase 1 latency budget.

Pick a non-default camera with `--webcam 1`, `--webcam 2`, etc.

### Equivalent direct invocation

`scripts/webcam_demo.py` is a thin wrapper around the same code path
that defaults the camera index to `0`:

```bash
python scripts/webcam_demo.py            # camera 0
python scripts/webcam_demo.py --cam 1    # camera 1
```

### Troubleshooting

- **`could not open webcam at index N`** — no camera at that index. Try
  another index, or check whether another process (browser tab, video
  call) has the device open.
- **`models directory not found`** — re-run
  `python scripts/download_models.py all --out app/models/`.
- **`prompts file not found`** — run the command from inside the repo
  checkout so the loader can find `docs/prompts.md`. In the container,
  `/config/prompts.yaml` is bind-mounted; set `PROMPTS_FILE` to override.

## Tests

```bash
pytest
```

Runs the worker test suite. The Phase 1 webcam demo is a manual gate, not
an automated test — the suite covers the pipeline pieces it composes.
