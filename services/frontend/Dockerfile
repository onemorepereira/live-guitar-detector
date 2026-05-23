# Frontend assets builder.
#
# Phase 3.1: produces a multi-stage image whose final stage exposes only
# the built static SPA at /dist. The gateway image (Task 3.2) uses this
# image as a build context and COPYs /dist into its own /app/static.
#
# This image is NEVER run as a container — it has no entrypoint.

# ---- build stage ----
FROM node:20.11-alpine AS build

WORKDIR /app

# Cache npm install across rebuilds when only source changed.
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

# Now copy source and build.
COPY tsconfig.json vite.config.ts tailwind.config.js postcss.config.js index.html ./
COPY src ./src
COPY tests ./tests
RUN npm run build

# ---- export stage ----
# Scratch with only the built dist/. Consumed via `COPY --from=...` in
# downstream images.
FROM scratch AS export
COPY --from=build /app/dist /dist
