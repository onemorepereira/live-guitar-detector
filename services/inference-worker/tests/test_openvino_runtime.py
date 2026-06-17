"""Tests for the pure OpenVINO compile-config helper."""

from __future__ import annotations

from app.openvino_runtime import compile_config


def test_compile_config_auto_threads_is_empty() -> None:
    """0 = auto → no INFERENCE_NUM_THREADS property (let OpenVINO decide)."""
    assert compile_config(0) == {}


def test_compile_config_pins_thread_count() -> None:
    """A positive thread count is passed through as a CPU runtime property."""
    assert compile_config(4) == {"INFERENCE_NUM_THREADS": "4"}
