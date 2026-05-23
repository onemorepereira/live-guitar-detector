.PHONY: install lint test build-images push-images dev dev-down dev-logs help
.DEFAULT_GOAL := help

help:    ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-20s %s\n", $$1, $$2}'
install: ## Install all dev deps
	@echo "TODO: install"
lint:    ## Run linters across services
	@echo "TODO: lint"
test:    ## Run all unit tests
	@echo "TODO: test"
dev:     ## Run local dev stack (docker-compose: redis + gateway + worker)
	@echo "Starting dev stack — frontend should be started separately with:"
	@echo "    cd services/frontend && npm run dev"
	docker compose up --build
dev-down: ## Tear down the dev stack
	docker compose down
dev-logs: ## Tail dev stack logs
	docker compose logs -f
build-images: ## Build container images
	@echo "TODO: build-images"
push-images:  ## Push images to local registry
	@echo "TODO: push-images"
