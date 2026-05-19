# Inference Worker Scripts

Utility scripts for the `inference-worker` service. The headline script is
`download_models.py`, which fetches model weights and exports them to
OpenVINO IR so the worker can load them at startup.

## `download_models.py`

Downloads and exports the two models the worker uses:

| Model           | Purpose                                    | Weights source             | Export target     |
| --------------- | ------------------------------------------ | -------------------------- | ----------------- |
| `yolov8n-oiv7`  | Detector (class "Guitar" from OIv7)        | Ultralytics GitHub release | OpenVINO IR, FP32 |
| `MobileCLIP-S1` | Zero-shot classifier (image + text towers) | OpenCLIP / Hugging Face    | OpenVINO IR, FP16 |

> **Note on MobileCLIP-S0 vs S1.** `DESIGN.md` specifies MobileCLIP-S0, but
> S0 weights are **not** published in OpenCLIP's registry as of v2.32. The
> closest available variant is `MobileCLIP-S1` (slightly larger but still
> well within the latency budget — Task 1.10 will confirm). The script's
> `--clip-model` flag lets you override this if S0 becomes available later.

### Subcommands

```
python scripts/download_models.py --help
python scripts/download_models.py all --out app/models/
python scripts/download_models.py download-yolo --out app/models/raw/
python scripts/download_models.py export-yolo   --in app/models/raw/yolov8n-oiv7.pt --out app/models/
python scripts/download_models.py download-clip --out app/models/raw/
python scripts/download_models.py export-clip   --in app/models/raw/mobileclip-state.pt --out app/models/
```

Pass `--force` (before the subcommand) to overwrite existing outputs. Without
it, each step skips if the expected file/directory already exists, so it is
safe to re-run.

### Setup

You need the extra ONNX dependencies (only used by the export step, not at
runtime):

```bash
cd services/inference-worker
source .venv/bin/activate
pip install -e .[export]
```

### Run

```bash
cd services/inference-worker
python scripts/download_models.py all --out app/models/
```

### Expected output tree

After a successful run:

```
app/models/
├── raw/                          # source .pt weights (kept for re-export)
│   ├── yolov8n-oiv7.pt           # ~7 MB
│   └── mobileclip-state.pt       # ~325 MB (state dict + metadata)
├── yolov8n-oiv7-fp32/
│   ├── yolov8n-oiv7.xml          # OpenVINO IR graph
│   ├── yolov8n-oiv7.bin          # ~13 MB weights
│   ├── metadata.yaml             # Ultralytics-emitted class names etc.
│   └── precision.json
├── mobileclip-image-fp16/
│   ├── image.xml
│   ├── image.bin                 # ~41 MB
│   └── precision.json
└── mobileclip-text-fp16/
    ├── text.xml
    ├── text.bin                  # ~121 MB
    └── precision.json
```

`precision.json` records the actual precision the model was exported at,
along with any caveats (e.g. "INT8 deferred") and the source model identity
(`model_name`, plus `pretrained` for the CLIP towers). Loaders should treat
the `precision` field as authoritative rather than parsing directory names.

The marker is also the **completion sentinel**: it is the last file written
on each successful export, so an export dir without a `precision.json`
(e.g. left behind by a crash or Ctrl-C in mid-flight) is treated as
incomplete and re-exported on the next run.

### Resource cost (observed)

| Metric                     | Value                                                            |
| -------------------------- | ---------------------------------------------------------------- |
| Wall-clock, full run       | ~30 s on a Ryzen 7940HS with warm pip cache and ~50 MB/s network |
| First-run network          | ~7 MB (YOLO) + ~325 MB (MobileCLIP) = **~332 MB** download       |
| Peak disk usage during run | ~750 MB (intermediate ONNX + IR coexist briefly)                 |
| Final disk usage           | ~510 MB (`raw/` plus three IR dirs)                              |

The original plan estimated ~2 GB / 5–10 minutes. Actual numbers are
substantially smaller because (a) we use MobileCLIP-S1 not a larger
encoder, and (b) we don't carry an INT8 calibration dataset on disk.

### Known follow-ups

- **YOLO INT8 quantization** — deferred. Ultralytics' `int8=True` export
  needs a calibration dataset YAML aligned with the model's class taxonomy.
  `coco128.yaml` (the default) does not cover OIv7 classes, so calibration
  would be meaningless. A follow-up task should either ship a small OIv7
  calibration subset or wire up NNCF/POT against representative camera
  frames once we have them. The `--yolo-int8` flag is wired up but will
  emit a warning and is likely to fall back to FP32.
- **CLIP image-tower INT8** — deferred for the same reason: needs a
  calibration dataset (Apple's MobileCLIP repo ships one, but it is not
  exposed through OpenCLIP). For now both towers ship at FP16. The
  `--clip-image-int8` flag is reserved for the follow-up.
- **MobileCLIP-S0** — re-evaluate once weights land in OpenCLIP (or wire
  in Apple's `ml-mobileclip` repo directly).

### Implementation notes

- `torch.onnx.export` now uses the dynamo path by default in torch 2.12; it
  emits a couple of harmless `Unable to load package` warnings during
  export (it probes the output file as a `.pt2` archive, fails, and falls
  back to the ONNX path). These can be ignored.
- The CLIP `.onnx` files (and any `*.onnx.data` external-data sidecars)
  are deleted automatically after the IR is written.
- `precision.json` in each export dir is a small JSON marker; the
  inference worker can read it at load time to pick the right OpenVINO
  config (e.g. INT8 needs different cache hints).
