# Deployment

Cluster manifests live in the **separate** `k3s` repo
(`~/Extra/repos/personal/k3s/`) under `cluster/guitar-detect/`. That
repo is the single source of truth for the Helm chart, namespace,
HelmRelease, and Flux Kustomization — this repo only builds and pushes
the container images.

Deployment is GitOps via FluxCD: commit to the k3s repo → Flux
reconciles within ~30 min (or `flux reconcile kustomization
guitar-detect` to force).

## First-time install

```bash
# 1. Build + push images to the in-cluster registry
make build-images TAG=0.1.0
make push-images  TAG=0.1.0
# REGISTRY defaults to registry.home.devoops.co; override with
#   make push-images REGISTRY=ghcr.io/me TAG=0.1.0

# 2. Train the SigLIP probe locally if you don't have one yet
cd services/inference-worker && source .venv/bin/activate
python scripts/train_probe.py --backend siglip \
  --data-dir ./data_crops \
  --out ./app/models/classifier-probe/probe_siglip.npz

# 3. Sync the probe artifact into the k3s repo's chart
cp services/inference-worker/app/models/classifier-probe/probe_siglip.npz \
   ~/Extra/repos/personal/k3s/cluster/guitar-detect/chart/files/probe_siglip.npz

# 4. Commit the k3s repo
cd ~/Extra/repos/personal/k3s
git add cluster/guitar-detect cluster/infrastructure/flux-kustomizations.yml
git commit -m "feat(guitar-detect): add app"
git push

# 5. Wait for reconciliation (or force it)
flux reconcile kustomization guitar-detect

# 6. Open https://guitars.home.devoops.co on a phone or desktop.
```

## Upgrade

```bash
make build-images TAG=0.2.0
make push-images  TAG=0.2.0

# In the k3s repo, bump values.image.tag in helmrelease.yml:
cd ~/Extra/repos/personal/k3s
# edit cluster/guitar-detect/helmrelease.yml -> values.image.tag: "0.2.0"
git commit -am "chore(guitar-detect): bump to 0.2.0"
git push
```

Rolling upgrade: gateway is 1 replica (brief outage during pod swap);
inference is 1 replica (sessions detect the gap and reconnect within
~10 s).

## Retraining the probe

The probe is shipped as a binary ConfigMap rendered from
`chart/files/probe_siglip.npz` in the k3s repo. To roll out a new probe:

```bash
# Retrain locally per services/inference-worker/scripts/TRAIN_PROBE.md.
# Then sync + commit:
cp services/inference-worker/app/models/classifier-probe/probe_siglip.npz \
   ~/Extra/repos/personal/k3s/cluster/guitar-detect/chart/files/probe_siglip.npz
cd ~/Extra/repos/personal/k3s
git commit -am "chore(guitar-detect): retrain SigLIP probe"
git push
```

Flux re-renders the ConfigMap, the worker Deployment picks up the new
mount on the next pod rollout (force one with `kubectl -n guitar-detect
rollout restart deploy/guitar-detect-inference`).

## Rollback

```bash
# Revert the k3s-repo commit:
cd ~/Extra/repos/personal/k3s
git revert <commit>
git push
flux reconcile kustomization guitar-detect
```

Or for a fast in-cluster rollback without touching git:

```bash
helm history guitar-detect -n guitar-detect
helm rollback guitar-detect <REV> -n guitar-detect
# (Flux will reconcile back to the git state on the next interval.)
```

## Values reference

The chart's `values.yaml` lives at `cluster/guitar-detect/chart/values.yaml`
in the k3s repo. Per-cluster overrides go in `cluster/guitar-detect/helmrelease.yml`
under `spec.values`. Common keys:

| Key                                 | Default                    | Override when                                                                                               |
| ----------------------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `image.tag`                         | `0.1.0`                    | Bumping releases                                                                                            |
| `image.registry`                    | `registry.home.devoops.co` | Using a different registry                                                                                  |
| `redis.image`                       | `redis:7-alpine`           | Pinning a specific Redis patch                                                                              |
| `ingress.host`                      | `guitars.home.devoops.co`  | Using a different hostname                                                                                  |
| `ingress.tls.issuer`                | `letsencrypt-prod`         | Using a different cert-manager ClusterIssuer                                                                |
| `inference.replicas`                | `1`                        | (Future) horizontal scaling                                                                                 |
| `inference.env.CLASSIFIER_MODE`     | `siglip_probe`             | Falling back to `zero_shot` or `probe`                                                                      |
| `inference.env.DETECT_CONF`         | `0.35`                     | Lower → more sensitive YOLO (catches more borderline frames; more spurious dets — ByteTrack's vote absorbs) |
| `inference.probe.enabled`           | `true`                     | Deploying without a probe (e.g. zero-shot mode)                                                             |
| `coturn.enabled`                    | `true`                     | Disable only if you have a different TURN relay or your phones can reach the gateway directly (rare in K8s) |
| `coturn.loadBalancerIP`             | `192.168.86.7`             | A free IP in your MetalLB pool. coturn's `external-ip` is baked from this so the values MUST match.         |
| `coturn.username`/`coturn.password` | `guitar`/`change-me`       | LAN-only credentials. Override via `helmrelease.yml` `spec.values`. SOPS-encrypt if the repo isn't private. |
| `networkPolicies.enabled`           | `true`                     | Disabling for debugging                                                                                     |
| `networkPolicies.traefikNamespace`  | `kube-system`              | Upstream Traefik install in a dedicated namespace                                                           |

### Why the cluster needs coturn

The gateway runs aiortc in a pod with a `10.42.x.x` cluster-internal IP that
isn't routable from the LAN. Without a TURN relay, a phone's WebRTC stack
gathers ICE candidates that can never reach the gateway, and ICE never
completes. The chart ships a `coturn` Deployment + LoadBalancer Service
pinned to a MetalLB IP; the gateway advertises that IP to clients via
`GET /api/config` and uses it itself as an ICE relay. See `chart/templates/coturn.yaml`.

## Operations runbook

### Watch live detection events for a session

```bash
SID=<session-id-from-browser-debug-panel>
kubectl -n guitar-detect exec -it sts/guitar-detect-redis -- \
  redis-cli XREAD BLOCK 0 STREAMS detections:${SID} '$'
```

### Check frame backlog

```bash
kubectl -n guitar-detect exec -it sts/guitar-detect-redis -- \
  redis-cli XLEN frames:${SID}
```

### Stream worker logs

```bash
kubectl -n guitar-detect logs -l app.kubernetes.io/component=inference \
  -f --tail=100
```

### Force session cleanup

```bash
kubectl -n guitar-detect exec -it deploy/guitar-detect-gateway -- \
  curl -X DELETE http://localhost:8000/api/session/${SID}
```

## What's intentionally NOT here

- Multi-viewer support (would need an SFU like mediasoup or Janus).
- Cross-session guitar identity ("this is the same Les Paul as yesterday")
  — needs embedding-based instance matching, not class labels.
- Detection event persistence to a database.
- Auth beyond LAN trust.
- GPU support — architecture is CPU-only by design.
