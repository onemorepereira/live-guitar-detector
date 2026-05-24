# Deployment

Target: 2-node home K3s cluster. One node labeled `workload=io` (gateway, redis, registry), one node `workload=compute` (inference).

See [`deploy/k3s/README.md`](../deploy/k3s/README.md) for cluster
bootstrap (registry install, mkcert TLS, root CA install per device, DNS).

## First-time install

```bash
# 0. Cluster prereqs (see deploy/k3s/README.md):
#    - K3s ≥ 1.28, Longhorn as default storage class
#    - kubectl context pointed at the cluster
#    - mkcert installed on the operator host

# 1. Label nodes
MOBILE_NODE=mobile COMPUTE_NODE=ryzen \
  ./deploy/k3s/label-nodes.sh

# 2. Local registry (one node hosts it via hostNetwork:5000)
./deploy/k3s/install-registry.sh
# Follow printed instructions:
#   - Add /etc/rancher/k3s/registries.yaml to every node + restart k3s.
#   - Add `<io-node-ip> registry.local` to /etc/hosts on every node and
#     dev host.

# 3. Build + push images
make build-images TAG=0.1.0
make push-images  TAG=0.1.0

# 4. TLS cert
./deploy/k3s/install-mkcert-cert.sh
# Then install $(mkcert -CAROOT)/rootCA.pem on every viewing device per
# the README's per-platform instructions.

# 5. DNS — find the ingress IP:
kubectl -n kube-system get svc traefik \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
# Add `<ingress-ip> guitars.home.lan` to your router DNS or /etc/hosts.

# 6. Deploy with Helm
helm install guitar-detect deploy/helm/guitar-detect \
  -f deploy/helm/guitar-detect/values.local.yaml \
  --create-namespace --namespace guitar-detect

# 7. Smoke test
./deploy/k3s/smoke-test.sh

# 8. Open https://guitars.home.lan on a phone or desktop.
```

## Upgrade

```bash
make build-images TAG=0.2.0
make push-images  TAG=0.2.0

helm upgrade guitar-detect deploy/helm/guitar-detect \
  -f deploy/helm/guitar-detect/values.local.yaml \
  --set image.tag=0.2.0 \
  --namespace guitar-detect
```

Rolling upgrade: gateway is 1 replica (brief outage during pod swap);
inference is 1 replica (sessions detect the gap and reconnect within
~10 s — see Test 9 in [E2E_CHECKLIST.md](E2E_CHECKLIST.md)).

## Rollback

```bash
helm history guitar-detect -n guitar-detect          # list revisions
helm rollback guitar-detect <REVISION> -n guitar-detect
```

## Values reference

`deploy/helm/guitar-detect/values.yaml` is the schema. `values.local.yaml`
is the per-cluster override starter. Common overrides:

| Key                                | Default               | Override when                                     |
| ---------------------------------- | --------------------- | ------------------------------------------------- |
| `image.tag`                        | `0.1.0`               | Bumping releases                                  |
| `image.registry`                   | `registry.local:5000` | Using a different registry                        |
| `redis.image`                      | `redis:7-alpine`      | Pinning a specific Redis patch                    |
| `ingress.host`                     | `guitars.home.lan`    | Using a different hostname                        |
| `inference.replicas`               | `1`                   | (Future) horizontal scaling                       |
| `networkPolicies.enabled`          | `true`                | Disabling for debugging                           |
| `networkPolicies.traefikNamespace` | `kube-system`         | Upstream Traefik install in a dedicated namespace |

## Operations runbook

### Watch live detection events for a session

```bash
SID=<session-id-from-browser-debug-panel>
kubectl -n guitar-detect exec -it sts/<redis-pod> -- \
  redis-cli XREAD BLOCK 0 STREAMS detections:${SID} '$'
```

### Check frame backlog

```bash
kubectl -n guitar-detect exec -it sts/<redis-pod> -- \
  redis-cli XLEN frames:${SID}
```

### Stream worker logs

```bash
kubectl -n guitar-detect logs -l app.kubernetes.io/component=inference \
  -f --tail=100
```

### Force session cleanup

```bash
kubectl -n guitar-detect exec -it deploy/<gateway-pod> -- \
  curl -X DELETE http://localhost:8000/api/session/${SID}
```

## What's intentionally NOT here

- Multi-viewer support (would need an SFU like mediasoup or Janus).
- Cross-session guitar identity ("this is the same Les Paul as yesterday")
  — needs embedding-based instance matching, not class labels.
- Detection event persistence to a database.
- Auth beyond LAN trust.
- GPU support — architecture is CPU-only by design.
