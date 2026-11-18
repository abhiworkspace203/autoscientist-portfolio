import os
from unittest.mock import MagicMock, patch
import pytest
from requests.exceptions import ReadTimeout
from docker.errors import DockerException

from src.sandbox.docker_runner import (
    execute_code_in_sandbox,
    is_docker_available,
    IMAGE_NAME,
)


# ---------------------------------------------------------------------------
# Live Integration Tests (executed if Docker daemon is active)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not running / reachable")
def test_sandbox_blocks_network():
    code = """
import urllib.request

try:
    urllib.request.urlopen("https://example.com", timeout=3)
except Exception:
    print("NETWORK_BLOCKED")
else:
    print("NETWORK_ALLOWED")
"""
    result = execute_code_in_sandbox(code)
    assert result["timed_out"] is False
    assert "NETWORK_BLOCKED" in result["stdout"]
    assert "NETWORK_ALLOWED" not in result["stdout"]


@pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not running / reachable")
def test_sandbox_stops_code_after_timeout():
    result = execute_code_in_sandbox(
        "import time; time.sleep(5)",
        timeout_sec=2,
    )
    assert result["timed_out"] is True
    assert result["exit_code"] == -1


# ---------------------------------------------------------------------------
# Unit & Security Mock Tests (verify Docker constraints independently of daemon)
# ---------------------------------------------------------------------------

def test_is_docker_available_returns_bool():
    available = is_docker_available()
    assert isinstance(available, bool)


@patch("src.sandbox.docker_runner.docker.from_env")
def test_sandbox_enforces_strict_resource_limits(mock_from_env):
    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_from_env.return_value = mock_client
    mock_client.containers.run.return_value = mock_container
    mock_container.wait.return_value = {"StatusCode": 0}
    mock_container.logs.side_effect = [b"baseline output\n", b""]

    test_code = "print('Simulating experiment')"
    result = execute_code_in_sandbox(test_code, timeout_sec=15)

    mock_client.containers.run.assert_called_once_with(
        image=IMAGE_NAME,
        command=["python", "-c", test_code],
        detach=True,
        network_mode="none",
        mem_limit="1g",
        cpu_quota=100_000,
        cpu_period=100_000,
        user="sandboxuser",
    )
    assert result["exit_code"] == 0
    assert result["timed_out"] is False
    assert "baseline output" in result["stdout"]
    mock_container.remove.assert_called_once_with(force=True)
    mock_client.close.assert_called_once()


@patch("src.sandbox.docker_runner.docker.from_env")
def test_sandbox_handles_timeout_and_kills_container(mock_from_env):
    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_from_env.return_value = mock_client
    mock_client.containers.run.return_value = mock_container
    mock_container.wait.side_effect = ReadTimeout("Timeout reached")
    mock_container.logs.side_effect = [b"partial logs", b"killed"]

    result = execute_code_in_sandbox("while True: pass", timeout_sec=2)

    mock_container.kill.assert_called_once()
    assert result["timed_out"] is True
    assert result["exit_code"] == -1
    mock_container.remove.assert_called_once_with(force=True)
    mock_client.close.assert_called_once()


@patch("src.sandbox.docker_runner.docker.from_env")
def test_sandbox_handles_docker_exception_gracefully(mock_from_env):
    mock_from_env.side_effect = DockerException("Docker daemon connection failed")

    result = execute_code_in_sandbox("print(1)")

    assert result["exit_code"] == -1
    assert result["timed_out"] is False
    assert "Docker error" in result["stderr"]
    assert "Docker daemon connection failed" in result["stderr"]


def test_dockerfile_specification():
    dockerfile_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "sandbox", "Dockerfile"
    )
    assert os.path.exists(dockerfile_path), "sandbox/Dockerfile must exist"

    with open(dockerfile_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "python:3.10-slim" in content
    assert "sandboxuser" in content
    assert "WORKDIR /workspace" in content
    assert "USER sandboxuser" in content
    for pkg in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]:
        assert pkg in content
