# K3s Deployment

Bootstrap and operations guide for the Guitar Detect home K3s cluster.

This document covers cluster-level setup (registry, mkcert, node labels)
that runs ONCE per cluster. Application deployment via Helm lives in
`deploy/helm/guitar-detect/` (see [DEPLOYMENT.md](../../docs/DEPLOYMENT.md)
when it lands).

## Prerequisites

- K3s 1.28+ on at least 2 nodes.
- One node labeled `workload=io` (gateway + redis + registry).
- One node labeled `workload=compute` (inference worker).
- Longhorn installed as the default storage class.
- `kubectl` configured on the operator host with admin context.
- `mkcert` installed on the operator host.

## One-time setup

Run these in order. All scripts are idempotent.

### 1. Label nodes

```bash
MOBILE_NODE=my-mobile-node COMPUTE_NODE=my-ryzen-node \
  ./deploy/k3s/label-nodes.sh
```

### 2. Install the private registry

```bash
./deploy/k3s/install-registry.sh
```

Then follow the printed instructions to:

- Write `/etc/rancher/k3s/registries.yaml` on each K3s node.
- Add `<io-node-ip> registry.local` to `/etc/hosts` on every node AND
  every dev host that will push images.
- Restart `k3s` / `k3s-agent` on each node.

Verify:

```bash
curl http://registry.local:5000/v2/    # → {}
```

### 3. Generate and install the TLS cert

```bash
./deploy/k3s/install-mkcert-cert.sh
```

This generates a mkcert-signed cert for `guitars.home.lan` and installs
it as a `guitars-tls` secret in the `guitar-detect` namespace.

### 4. Install the mkcert root CA on each viewing device

Locate the root CA:

```bash
mkcert -CAROOT
# e.g. /home/you/.local/share/mkcert
```

Then copy `rootCA.pem` from that directory to each device and install it
per the device's platform:

#### macOS

```bash
sudo security add-trusted-cert -d -r trustRoot \
  -k /Library/Keychains/System.keychain rootCA.pem
```

Or open `Keychain Access`, drag `rootCA.pem` into **System**, and set
the trust to **Always Trust**.

#### iOS / iPadOS

1. AirDrop or email `rootCA.pem` to the device.
2. Open the profile and tap **Install** (twice, with passcode).
3. Then go to **Settings → General → About → Certificate Trust Settings**
   and enable full trust for the **mkcert** root.

#### Android

1. Copy `rootCA.pem` to the device's **Downloads** folder.
2. Settings → **Security & privacy** (or **Security**) → **Encryption &
   credentials** → **Install a certificate** → **CA certificate**.
3. Accept the security warning, browse to `rootCA.pem`, select it.

Newer Android (10+) restricts user-installed CAs to _user trust only_ —
Chrome respects it; some apps that pin to the system store will not.

#### Linux (Chrome / Chromium / Edge)

Chrome reads NSS:

```bash
sudo apt install libnss3-tools     # if not already
mkdir -p $HOME/.pki/nssdb
certutil -A -d sqlite:$HOME/.pki/nssdb -n mkcert-root \
  -i $(mkcert -CAROOT)/rootCA.pem -t "TC,Cw,Tw"
```

Restart Chrome.

#### Linux (Firefox)

Firefox has its own trust store. About → **Settings** → **Privacy &
Security** → scroll to **Certificates** → **View Certificates** →
**Authorities** → **Import**, select `rootCA.pem`, check **Trust this
CA to identify websites**.

### 5. DNS — make `guitars.home.lan` resolvable

Find the K3s ingress IP:

```bash
kubectl -n kube-system get svc traefik \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

Then either:

- **Router DNS** — add an A record for `guitars.home.lan` pointing at
  that IP. Recommended for shared LAN access.
- **/etc/hosts on each device** — add a line:
  ```
  <ingress-ip>  guitars.home.lan
  ```

### 6. Deploy the application

Once all of the above are done, install the Helm chart:

```bash
helm install guitar-detect deploy/helm/guitar-detect \
  -f deploy/helm/guitar-detect/values.local.yaml \
  --create-namespace --namespace guitar-detect
```

(See `deploy/helm/guitar-detect/` once Task 4.4 lands.)

## Troubleshooting

| Symptom                                                           | Cause                                                              | Fix                                                     |
| ----------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------- |
| `curl https://guitars.home.lan` → cert warning on dev host        | Root CA not installed on this device                               | Install rootCA.pem per platform (above)                 |
| iOS/Android camera picker is empty                                | Same — `getUserMedia` requires HTTPS or trusted cert               | Install root CA AND ensure DNS resolves                 |
| `kubectl apply -f tls-secret` succeeds but ingress shows old cert | Traefik caches certs                                               | `kubectl -n kube-system rollout restart deploy traefik` |
| Registry pull `x509: certificate signed by unknown authority`     | K3s nodes need `tls.insecure_skip_verify: true` in registries.yaml | See `install-registry.sh` printout                      |
