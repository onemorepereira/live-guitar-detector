.PHONY: install lint test build-images push-images dev help
.DEFAULT_GOAL := help

help:    ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-20s %s\n", $$1, $$2}'
install: ## Install all dev deps
	@echo "TODO: install"
lint:    ## Run linters across services
	@echo "TODO: lint"
test:    ## Run all unit tests
	@echo "TODO: test"
dev:     ## Run local dev stack (docker-compose)
	@echo "TODO: dev"
build-images: ## Build container images
	@echo "TODO: build-images"
push-images:  ## Push images to local registry
	@echo "TODO: push-images"
