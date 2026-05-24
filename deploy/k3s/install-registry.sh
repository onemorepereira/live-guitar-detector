#!/usr/bin/env bash
# Deploy a private container registry for the Guitar Detect K3s cluster.
#
# Result: `registry.local:5000` is reachable from K3s nodes and from the
# developer's dev host, allowing `docker push registry.local:5000/...`
# and pod image pulls from the same address.
#
# Idempotent: re-runs are safe; existing resources are updated in place
# via `kubectl apply`.
#
# Requirements:
#   - kubectl context pointed at the target cluster
#   - At least one node labeled `workload=io` (run deploy/k3s/label-nodes.sh first)
#   - Longhorn storage class available (or set STORAGE_CLASS env var)
#
# Post-install steps (printed at the end) for each K3s node:
#   1. Write /etc/rancher/k3s/registries.yaml with the mirror config.
#   2. systemctl restart k3s (server) / k3s-agent (agent).
#   3. Add `<io-node-ip> registry.local` to /etc/hosts on dev host(s).

set -euo pipefail

NAMESPACE="${REGISTRY_NAMESPACE:-registry}"
STORAGE_CLASS="${STORAGE_CLASS:-longhorn}"
STORAGE_SIZE="${STORAGE_SIZE:-5Gi}"
REGISTRY_IMAGE="${REGISTRY_IMAGE:-registry:2.8.3}"

echo "==> Creating namespace: ${NAMESPACE}"
kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1 || kubectl create namespace "${NAMESPACE}"

echo "==> Applying registry resources (PVC, Deployment, Service)"
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: registry-data
  namespace: ${NAMESPACE}
spec:
  accessModes: ["ReadWriteOnce"]
  storageClassName: ${STORAGE_CLASS}
  resources:
    requests:
      storage: ${STORAGE_SIZE}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: registry
  namespace: ${NAMESPACE}
  labels:
    app: registry
spec:
  replicas: 1
  strategy:
    type: Recreate          # PVC is RWO; no rolling update.
  selector:
    matchLabels:
      app: registry
  template:
    metadata:
      labels:
        app: registry
    spec:
      nodeSelector:
        workload: io
      hostNetwork: true     # Listens on \${io-node-ip}:5000.
      dnsPolicy: ClusterFirstWithHostNet
      containers:
        - name: registry
          image: ${REGISTRY_IMAGE}
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 5000
              hostPort: 5000
              name: http
          env:
            - name: REGISTRY_HTTP_ADDR
              value: ":5000"
            - name: REGISTRY_STORAGE_FILESYSTEM_ROOTDIRECTORY
              value: "/var/lib/registry"
          volumeMounts:
            - name: data
              mountPath: /var/lib/registry
          readinessProbe:
            httpGet:
              path: /v2/
              port: 5000
            initialDelaySeconds: 3
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /v2/
              port: 5000
            initialDelaySeconds: 30
            periodSeconds: 30
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: registry-data
---
apiVersion: v1
kind: Service
metadata:
  name: registry
  namespace: ${NAMESPACE}
  labels:
    app: registry
spec:
  selector:
    app: registry
  ports:
    - name: http
      port: 5000
      targetPort: 5000
EOF

echo "==> Waiting for registry to become ready"
kubectl -n "${NAMESPACE}" rollout status deploy/registry --timeout=120s

NODE_IP=$(kubectl get nodes -l workload=io \
  -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')

if [[ -z "${NODE_IP}" ]]; then
  echo "WARNING: could not detect an InternalIP for any workload=io node." >&2
  NODE_IP="<workload-io-node-ip>"
fi

cat <<EOF

============================================================================
Registry deployed.

The registry is running with hostNetwork on the workload=io node and is
reachable at:    http://${NODE_IP}:5000

To make 'registry.local:5000' resolvable cluster-wide and from the dev
host, do the following ONCE per K3s node and ONCE on every dev host:

1. K3s nodes — create /etc/rancher/k3s/registries.yaml with:

     mirrors:
       "registry.local:5000":
         endpoint:
           - "http://registry.local:5000"
     configs:
       "registry.local:5000":
         tls:
           insecure_skip_verify: true

   Then:
     sudo chmod 644 /etc/rancher/k3s/registries.yaml
     sudo systemctl restart k3s          # k3s server nodes
     sudo systemctl restart k3s-agent    # k3s agent nodes

2. Every host (K3s nodes AND your dev machine) — add to /etc/hosts:

     ${NODE_IP}  registry.local

3. Verify from the dev host:

     curl http://registry.local:5000/v2/

   Expected: {} (empty JSON object, HTTP 200)

4. Push your first images:

     make build-images TAG=0.1.0
     make push-images  TAG=0.1.0
============================================================================
EOF
