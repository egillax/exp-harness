from __future__ import annotations

import os

import pytest

from tests.helpers import clear_offline_env, docker_available


@pytest.fixture(autouse=True)
def sanitize_env(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_offline_env(monkeypatch)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_docker = bool(os.environ.get("RUN_EXPERIMENT_TEST_DOCKER"))
    run_slow = bool(os.environ.get("RUN_EXPERIMENT_TEST_SLOW"))

    for item in items:
        if item.get_closest_marker("docker") and (not run_docker or not docker_available()):
            item.add_marker(
                pytest.mark.skip(
                    reason="docker tests are gated; set RUN_EXPERIMENT_TEST_DOCKER=1 and require docker"
                )
            )
        if item.get_closest_marker("slow") and not run_slow:
            item.add_marker(
                pytest.mark.skip(reason="slow tests are gated; set RUN_EXPERIMENT_TEST_SLOW=1")
            )
