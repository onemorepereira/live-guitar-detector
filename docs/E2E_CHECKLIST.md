# End-to-End Test Checklist

Manual gate for each phase. Run from a host with the cluster's TLS cert
trusted (mkcert root CA installed) and `guitars.home.lan` resolvable.

| #   | Test                                                                       | Pass criteria                                                                    |
| --- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| 1   | Open the landing page on **desktop Chrome**                                | Camera picker populated, mkcert green padlock                                    |
| 2   | Same on **iOS Safari**                                                     | Same behavior (mkcert CA installed on the device)                                |
| 3   | Point at a **Stratocaster** (use a printed reference if you don't own one) | Stable `Fender Stratocaster` within 2s of view                                   |
| 4   | Point at a **Les Paul**                                                    | Stable `Gibson Les Paul` within 2s of view                                       |
| 5   | Point at an **acoustic guitar**                                            | `Analyzing…` then stays unlabeled (Unknown rejection prompt wins)                |
| 6   | Walk a guitar **out of frame, back in**                                    | Same track ID if <2s, new ID if >2s (per DESIGN.md §1.4 track persistence)       |
| 7   | Two guitars in frame at once                                               | Two independent track IDs and two independent labels                             |
| 8   | Toggle wifi off then back on on the viewing device                         | UI shows "reconnecting", recovers within ~5s                                     |
| 9   | **Kill the inference pod mid-session** (see procedure below)               | UI continues serving video, detections resume within ~10s                        |
| 10  | **Close the browser tab** with the session active                          | Gateway logs show session teardown within `SESSION_IDLE_TIMEOUT_S` (default 10s) |

## Test 9: pod kill resilience procedure

While the app is running and locked onto a guitar on the viewing device:

1. On the operator dev host:
   ```bash
   kubectl -n guitar-detect delete pod -l app.kubernetes.io/component=inference
   ```
2. Observe the viewing device:
   - The live video keeps streaming (gateway is unaffected).
   - The HUD overlay freezes briefly — the last detection event still drawn.
   - Within ~10s a new inference pod becomes Ready (`kubectl get pods -n guitar-detect -w`), drains the in-flight frames from `frames:{session_id}`, and detection events resume.
   - The HUD lock-on re-acquires (track ID may change — ByteTrack restarts cold in the new pod).

Expected total recovery: ≤ 15s (10s pod startup + 5s ByteTrack warm-up).

If recovery exceeds 30s:

- `kubectl describe pod -n guitar-detect -l app.kubernetes.io/component=inference` — check for ImagePullBackOff or readiness probe failures.
- `kubectl logs -n guitar-detect -l app.kubernetes.io/component=inference --tail=100` — look for model load errors.
- Confirm `/tmp/ready` is touched after pipeline init (`grep -n /tmp/ready services/inference-worker/app/main.py`).

## Phase-by-phase gates

- **Phase 2** (dev compose): tests 1, 3, 4, 5, 6, 7. No HTTPS yet (localhost is treated as secure context).
- **Phase 4** (K3s install): tests 1-10. All gates must pass before declaring the deploy stable.
- **Phase 5** (polish): re-run 1-10 plus visual polish acceptance (HUD animation, gallery panel).
