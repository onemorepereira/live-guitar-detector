"""Inference worker package.

See DESIGN.md §5 for the worker contract: consumes JPEG frames from Redis
Streams, runs YOLO detection + MobileCLIP zero-shot classification, applies
rolling-window voting per track, and publishes annotated results back to Redis.
"""

__version__ = "0.1.0"
