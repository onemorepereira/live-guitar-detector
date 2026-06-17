"""Shared OpenVINO runtime helpers.

Keeps the env-var → ``compile_model`` config translation in one pure,
unit-testable place so both OpenVINO-backed classifiers apply
``OPENVINO_THREADS`` identically.
"""

from __future__ import annotations


def compile_config(num_threads: int) -> dict[str, str]:
    """Build the ``compile_model`` config dict for a CPU thread cap.

    ``num_threads <= 0`` means "auto" — return an empty dict so OpenVINO picks
    its own default. A positive value pins ``INFERENCE_NUM_THREADS`` (the CPU
    plugin property) so a worker on a shared node doesn't oversubscribe cores.
    """
    if num_threads and num_threads > 0:
        return {"INFERENCE_NUM_THREADS": str(num_threads)}
    return {}
