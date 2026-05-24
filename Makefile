.PHONY: install lint test test-integration build build-images push-images dev dev-down dev-logs prod prod-down help
.DEFAULT_GOAL := help

# Override on the command line to tag a release: `make build-images TAG=1.2.3`.
TAG ?= 0.1.0

help:    ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-20s %s\n", $$1, $$2}'
install: ## Install all dev deps
	@echo "TODO: install"
lint:    ## Run linters across services
	@echo "TODO: lint"
test:    ## Run all unit tests
	@echo "TODO: test"
test-integration: ## Run the integration test harness (synthetic frames -> gateway -> worker -> WS)
	docker compose -f docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from harness
	docker compose -f docker-compose.test.yml down -v
dev:     ## Run local dev stack (docker-compose: redis + gateway + worker, hot-reload)
	@echo "Starting dev stack — frontend should be started separately with:"
	@echo "    cd services/frontend && npm run dev"
	docker compose --profile dev up --build
dev-down: ## Tear down the dev stack
	docker compose --profile dev down
dev-logs: ## Tail dev stack logs
	docker compose logs -f
prod:    ## Run prod docker-compose stack (built images, no source bind-mounts)
	@echo "Building frontend-assets image first (gateway COPYs --from this image)..."
	docker build services/frontend -t guitar-detect/frontend-assets:dev
	@echo "Starting prod stack..."
	docker compose --profile prod up --build
prod-down: ## Tear down the prod stack
	docker compose --profile prod down
build: build-images ## Alias for build-images
build-images: ## Build all 3 container images in dependency order (TAG=... overrides; default $(TAG))
	@echo "==> [1/3] frontend-assets:$(TAG)  (npm ci + vite build; ~30s)"
	docker build services/frontend -t guitar-detect/frontend-assets:$(TAG)
	@echo "==> [2/3] gateway:$(TAG)          (aiortc deps + frontend bundle; ~1-2 min cold)"
	docker build services/gateway \
		--build-arg FRONTEND_IMAGE=guitar-detect/frontend-assets:$(TAG) \
		-t guitar-detect/gateway:$(TAG)
	@echo "==> [3/3] inference-worker:$(TAG)  (torch + ultralytics + openvino + SigLIP-2 bake; ~5-10 min cold)"
	docker build services/inference-worker -t guitar-detect/inference-worker:$(TAG)
	@echo
	@echo "All 3 images built. Tag: $(TAG)"
	@echo "Push with:   make push-images TAG=$(TAG)"
	@echo "Run prod:    make prod"
push-images: ## Push container images to registry.local:5000 (override TAG=...)
	docker tag guitar-detect/gateway:$(TAG)          registry.local:5000/guitar-detect/gateway:$(TAG)
	docker tag guitar-detect/inference-worker:$(TAG) registry.local:5000/guitar-detect/inference-worker:$(TAG)
	docker push registry.local:5000/guitar-detect/gateway:$(TAG)
	docker push registry.local:5000/guitar-detect/inference-worker:$(TAG)
