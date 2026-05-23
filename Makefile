.PHONY: install lint test test-integration build-images push-images dev dev-down dev-logs prod prod-down help
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
build-images: ## Build all container images (override TAG=... for version, default $(TAG))
	docker build services/frontend -t guitar-detect/frontend-assets:$(TAG)
	docker build services/gateway \
		--build-arg FRONTEND_IMAGE=guitar-detect/frontend-assets:$(TAG) \
		-t guitar-detect/gateway:$(TAG)
	docker build services/inference-worker -t guitar-detect/inference-worker:$(TAG)
push-images:  ## Push images to local registry
	@echo "TODO: push-images"
