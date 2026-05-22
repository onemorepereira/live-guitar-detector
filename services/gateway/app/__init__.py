"""Gateway package.

See DESIGN.md §5 for the gateway contract: terminates the browser's WebRTC
ingest, decodes frames to JPEG, publishes them to Redis Streams for the
inference worker, and forwards detection results back to the browser over
a WebSocket.
"""

__version__ = "0.1.0"
