.PHONY: install lint test test-integration build build-images push-images dev dev-down dev-logs prod prod-down runtime help
.DEFAULT_GOAL := help

# Override on the command line to tag a release: `make build-images TAG=1.2.3`.
TAG ?= 0.1.0

# Container runtime detection.
#
# Prefer docker if present (most CI / cloud envs ship it); fall back to
# podman (Fedora workstations don't ship dockerd by default but do ship
# podman + a podman-compose shim). Either can be forced from the command
# line: `make build CONTAINER=podman` or `make dev COMPOSE='podman-compose'`.
#
# We resolve once at parse time so every recipe sees the same runtime,
# and so the chosen tool prints in `make runtime` for easy debugging.
ifeq ($(origin CONTAINER), undefined)
CONTAINER := $(shell command -v docker >/dev/null 2>&1 && echo docker || \
                     (command -v podman >/dev/null 2>&1 && echo podman || echo none))
endif
ifeq ($(origin COMPOSE), undefined)
COMPOSE := $(shell command -v docker >/dev/null 2>&1 && echo "docker compose" || \
                   (command -v podman >/dev/null 2>&1 && \
                    (command -v podman-compose >/dev/null 2>&1 && echo "podman-compose" || echo "podman compose") || \
                    echo "none"))
endif

# Bail early if neither runtime is installed.
_check_runtime:
	@if [ "$(CONTAINER)" = "none" ]; then \
	  echo "ERROR: neither 'docker' nor 'podman' is on PATH." >&2; \
	  echo "Install one of:" >&2; \
	  echo "  - Docker:  https://docs.docker.com/engine/install/" >&2; \
	  echo "  - Podman:  https://podman.io/docs/installation" >&2; \
	  exit 1; \
	fi

help:    ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-20s %s\n", $$1, $$2}'
runtime: ## Print the detected container runtime + compose command
	@echo "CONTAINER = $(CONTAINER)"
	@echo "COMPOSE   = $(COMPOSE)"
install: ## Install all dev deps
	@echo "TODO: install"
lint:    ## Run linters across services
	@echo "TODO: lint"
test:    ## Run all unit tests
	@echo "TODO: test"
test-integration: _check_runtime ## Run the integration test harness (synthetic frames -> gateway -> worker -> WS)
	$(COMPOSE) -f docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from harness
	$(COMPOSE) -f docker-compose.test.yml down -v
dev: _check_runtime ## Run local dev stack (redis + gateway + worker, hot-reload). Run frontend separately.
	@echo "Starting dev stack via $(COMPOSE) — frontend should be started separately with:"
	@echo "    cd services/frontend && npm run dev"
	$(COMPOSE) --profile dev up --build
dev-down: _check_runtime ## Tear down the dev stack
	$(COMPOSE) --profile dev down
dev-logs: _check_runtime ## Tail dev stack logs
	$(COMPOSE) logs -f
prod: _check_runtime ## Run prod docker-compose stack (built images, no source bind-mounts)
	@echo "Building frontend-assets image first (gateway COPYs --from this image)..."
	$(CONTAINER) build services/frontend -t guitar-detect/frontend-assets:dev
	@echo "Starting prod stack via $(COMPOSE)..."
	$(COMPOSE) --profile prod up --build
prod-down: _check_runtime ## Tear down the prod stack
	$(COMPOSE) --profile prod down
build: build-images ## Alias for build-images
build-images: _check_runtime ## Build all 3 container images in dependency order (TAG=... overrides; default $(TAG))
	@echo "==> Using $(CONTAINER) for build."
	@echo "==> [1/3] frontend-assets:$(TAG)  (npm ci + vite build; ~30s)"
	$(CONTAINER) build services/frontend -t guitar-detect/frontend-assets:$(TAG)
	@echo "==> [2/3] gateway:$(TAG)          (aiortc deps + frontend bundle; ~1-2 min cold)"
	$(CONTAINER) build services/gateway \
		--build-arg FRONTEND_IMAGE=guitar-detect/frontend-assets:$(TAG) \
		-t guitar-detect/gateway:$(TAG)
	@echo "==> [3/3] inference-worker:$(TAG)  (torch + ultralytics + openvino + SigLIP-2 bake; ~5-10 min cold)"
	$(CONTAINER) build services/inference-worker -t guitar-detect/inference-worker:$(TAG)
	@echo
	@echo "All 3 images built. Tag: $(TAG)"
	@echo "Push with:   make push-images TAG=$(TAG)"
	@echo "Run prod:    make prod"
push-images: _check_runtime ## Push container images to registry.local:5000 (override TAG=...)
	$(CONTAINER) tag guitar-detect/gateway:$(TAG)          registry.local:5000/guitar-detect/gateway:$(TAG)
	$(CONTAINER) tag guitar-detect/inference-worker:$(TAG) registry.local:5000/guitar-detect/inference-worker:$(TAG)
	$(CONTAINER) push registry.local:5000/guitar-detect/gateway:$(TAG)
	$(CONTAINER) push registry.local:5000/guitar-detect/inference-worker:$(TAG)
