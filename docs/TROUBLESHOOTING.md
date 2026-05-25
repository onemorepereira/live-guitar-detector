# Troubleshooting

## Common issues

(From DESIGN.md §10.3 plus lessons learned during implementation.)

| Symptom                          | Likely cause                                    | Fix                                                                                                                 |
| -------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Browser: no camera in picker     | Not on HTTPS                                    | Cluster ingress terminates TLS via cert-manager (Let's Encrypt). For local dev see [DEVELOPMENT.md](DEVELOPMENT.md) |
| Browser: WebRTC connection fails | Traefik not forwarding WS / UDP issues          | Check Traefik logs; confirm UDP not needed (aiortc TCP fallback)                                                    |
| No detections appearing          | Worker can't reach Redis OR no frames flowing   | `kubectl logs` worker; check `XLEN frames:*` in redis                                                               |
| Wrong classifications            | CLIP prompts off OR detector cropping wrong     | Tweak `docs/prompts.md`; verify bbox padding in `classifier.py`                                                     |
| Worker pod OOMKilled             | Model memory grew (multi-replica on same node?) | Increase limit OR reduce replicas                                                                                   |
| High latency                     | Inference falling behind                        | Check `frames_dropped_total`; reduce `MAX_INGEST_FPS` or `DETECT_IMGSZ`                                             |
| Pod scheduled on wrong node      | Node labels missing or mismatched               | Re-run `label-nodes.sh`, `kubectl describe node`                                                                    |

## Implementation-time gotchas

These didn't make it into the original spec but cost cycles during the
build.

### Worker

| Symptom                                                    | Cause                                                             | Fix                                                                                                                                              |
| ---------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `download_models.py` fails on `MobileCLIP-S0`              | S0 not in OpenCLIP registry                                       | Script falls back to S1 automatically; see `services/inference-worker/scripts/README.md`                                                         |
| `model.track()` raises ImportError for `lap` at runtime    | Ultralytics autoinstalls `lap` on first call                      | Pinned `lap>=0.5.12,<1` in worker `pyproject.toml` as a worker dep                                                                               |
| Idempotent re-runs of `download_models.py` skip everything | Sentinel-based caching                                            | Delete the relevant `precision.json` to force re-export of a single layer                                                                        |
| Worker test `requires_model` skips locally                 | Models not downloaded                                             | `python scripts/download_models.py all --out app/models/`                                                                                        |
| Accuracy tests skip even with models present               | Synthetic fixture marker                                          | `touch services/inference-worker/tests/fixtures/images/REAL.txt` after dropping real photos                                                      |
| HUD always shows "Analyzing…", never a brand/model         | `CLASSIFIER_MODE=zero_shot` and the rejection prompts are winning | Switch to `siglip_probe` (see [CLASSIFIER.md](CLASSIFIER.md)) and train a probe                                                                  |
| `siglip_probe` startup error: probe head missing           | Probe `.npz` not on disk                                          | `python scripts/train_probe.py --backend siglip ...` to generate one (see [TRAIN_PROBE.md](../services/inference-worker/scripts/TRAIN_PROBE.md)) |
| `siglip_probe` startup error: cannot load model from HF    | `TRANSFORMERS_OFFLINE=1` set but HF cache empty                   | Pre-warm the cache (image build does this automatically) or unset the env var temporarily                                                        |
| Worker pegs CPU when in `siglip_probe` mode                | SigLIP-2 inference is ~50–100ms/crop on CPU                       | Expected; the per-track classify scheduler keeps total throughput in budget. Drop to `probe` mode if absolutely starved for CPU                  |

### Gateway

| Symptom                                          | Cause                                                         | Fix                                                                    |
| ------------------------------------------------ | ------------------------------------------------------------- | ---------------------------------------------------------------------- |
| WS test occasionally hangs                       | TestClient + WebSocket lifecycle race                         | Use `asyncio.shield` in cleanup; see commit history                    |
| aiortc dev image fails apt-get on `libavcodec59` | Debian bookworm package names                                 | Confirm base is `python:3.11-slim` (bookworm); names hold              |
| Static-file mount path                           | gateway prod Dockerfile copies frontend dist to `/app/static` | If 404 on `/`, check `_STATIC_DIR.is_dir()` guard in `main.py`         |
| Session "vanishes" mid-stream                    | 60s sliding TTL on `session:{id}` Hash                        | Ensure WebRTC is publishing frames (touches `last_frame_ts`) at 30 FPS |

### Frontend

| Symptom                                     | Cause                                              | Fix                                                                                |
| ------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Camera not released after Stop              | `useCamera`'s cleanup only fires on full unmount   | Refresh the page or click Reset — proper fix is a Phase 5+ task                    |
| HUD doesn't draw on initial render          | `denormalizeBbox` needs both video + element sizes | Confirm `ResizeObserver` fires; `VideoStage` guards `elW > 0`                      |
| Rapid `select()` calls produce stale stream | Known race in `useCamera`                          | Avoid clicking the camera dropdown twice in <100ms; defensive fix is a future task |
| `?debug=1` panel shows zeros                | First WS message hasn't arrived yet                | Wait ~1s after Start; the 2s sliding window populates                              |

### WebRTC / TURN

| Symptom                                                            | Cause                                                                                                                                          | Fix                                                                                                                                                             |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phone shows `WebRTC: connecting` forever, peer closes after ~10 s  | Phone trying to reach the gateway's pod IP (`10.42.x.x`) directly — unreachable from LAN. TURN relay missing or misconfigured.                 | Confirm `coturn` pod is Running and the LoadBalancer Service has an IP. Test with: `nc -zvu <coturn-lb-ip> 3478`. Check `GET /api/config` returns `iceServers`. |
| Gateway logs `OFFER candidates: none yet`                          | Browser is using trickle ICE but we don't have a candidate-exchange channel.                                                                   | Frontend must wait for `iceGatheringState === "complete"` before posting the offer. See `services/frontend/src/hooks/useWebRTC.ts`.                             |
| coturn logs `CREATE_PERMISSION processed, error 403: Forbidden IP` | One of the gathered ICE candidates is in a range coturn refuses to relay to (commonly IPv6 link-local or the relay IP itself). Usually benign. | If ICE still completes via another pair, ignore. Otherwise add `allowed-peer-ip=<range>` to the coturn ConfigMap.                                               |

### Deployment / K3s

| Symptom                                                                                                                       | Cause                                                                                                             | Fix                                                                                                                            |
| ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Registry pod Pending                                                                                                          | Port 5000 in use OR Longhorn missing                                                                              | `ss -ltnp ':5000'` on the io-node; install Longhorn first                                                                      |
| NetworkPolicies silently no-op                                                                                                | K3s `--disable-network-policy` flag                                                                               | Re-enable network-policy controller or remove the flag                                                                         |
| Ingress 504 on `/ws`                                                                                                          | Traefik not handling WS                                                                                           | Confirm Traefik version ≥ 2.x; no extra annotation needed                                                                      |
| `helm install` hangs on namespace                                                                                             | Namespace was pre-created out of band                                                                             | Either `--set namespace.create=false` or delete the existing namespace                                                         |
| Image pull `x509: certificate signed by unknown authority`                                                                    | K3s nodes don't trust the registry (HTTP)                                                                         | Add `tls.insecure_skip_verify: true` to `/etc/rancher/k3s/registries.yaml`                                                     |
| `helm upgrade` fails with `StatefulSet ... is invalid: spec: Forbidden: updates to statefulset spec for fields other than...` | A label on a `volumeClaimTemplate` is drifting between renders (e.g. `helm.sh/chart` gets the git sha from Flux). | Use only selector labels (`name/instance/component`) on VCT metadata. Already fixed for redis in `chart/templates/redis.yaml`. |
| MetalLB Service stuck `EXTERNAL-IP <pending>`                                                                                 | Both `spec.loadBalancerIP` AND a `metallb.io/loadBalancerIPs` annotation set → MetalLB refuses to allocate.       | Use one or the other, not both. We use `spec.loadBalancerIP`.                                                                  |

### Visual UX

| Symptom                                                         | Cause                                                                                   | Fix                                                                                                                                                                   |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Bbox flickers on/off                                            | YOLO occasionally misses the target for 1–5 s; tracker re-acquires with a new track_id. | Tune `HOLD_MS`, `MIN_DRAW_AGE_FRAMES` in `services/frontend/src/components/HUD.tsx` and `DETECT_CONF` in the worker env (lower → more sensitive, more spurious dets). |
| Bbox jitters / wobbles                                          | YOLO per-frame position noise even on a static target.                                  | Tune `BBOX_EMA_ALPHA` in `HUD.tsx` (lower → smoother, more lag).                                                                                                      |
| `/` returns `{"detail": "Not Found"}` (JSON) instead of the SPA | Production image's site-packages-relative path doesn't find `/app/static`.              | `main.py` now checks `/app/static` first and falls back to `__file__`-relative for editable installs.                                                                 |

## Where logs live

- Gateway: `kubectl -n guitar-detect logs -l app.kubernetes.io/component=gateway`
- Worker: `kubectl -n guitar-detect logs -l app.kubernetes.io/component=inference`
- Redis: `kubectl -n guitar-detect logs <redis-pod-name>`
- Traefik: `kubectl -n kube-system logs -l app.kubernetes.io/name=traefik`

## Diagnostic CLI commands

See [DEPLOYMENT.md → Operations runbook](DEPLOYMENT.md#operations-runbook)
for the canonical kubectl snippets (frame backlog, force session cleanup,
log streams).

## Asking for help

Open an issue with:

- Phase you're in (dev compose, prod compose, K3s).
- Output of the relevant log command above.
- `helm get values guitar-detect -n guitar-detect` if K3s.
- Browser console + network tab if frontend.
