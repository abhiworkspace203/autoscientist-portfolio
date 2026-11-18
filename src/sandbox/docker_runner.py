from typing import Any

from requests.exceptions import ConnectionError as RequestConnectionError
from requests.exceptions import ReadTimeout
import docker
from docker.errors import DockerException


IMAGE_NAME = "scientific-sandbox:latest"


def is_docker_available() -> bool:
    """Check if Docker daemon is running and reachable."""
    try:
        client = docker.from_env()
        client.ping()
        client.close()
        return True
    except Exception:
        return False


def execute_code_in_sandbox(
    code_string: str,
    timeout_sec: int = 30,
) -> dict[str, Any]:
    """Run Python code in an isolated Docker container."""
    client = None
    container = None

    try:
        client = docker.from_env()
        container = client.containers.run(
            image=IMAGE_NAME,
            command=["python", "-c", code_string],
            detach=True,
            network_mode="none",
            mem_limit="1g",
            cpu_quota=100_000,
            cpu_period=100_000,
            user="sandboxuser",
        )

        try:
            result = container.wait(timeout=timeout_sec)
            timed_out = False
            exit_code = result.get("StatusCode", 1)
        except (ReadTimeout, RequestConnectionError):
            timed_out = True
            exit_code = -1

        if timed_out:
            try:
                container.kill()
            except Exception:
                pass
            stdout = container.logs(stdout=True, stderr=False).decode(
                "utf-8", errors="replace"
            )
            stderr = container.logs(stdout=False, stderr=True).decode(
                "utf-8", errors="replace"
            )
        else:
            stdout = container.logs(stdout=True, stderr=False).decode(
                "utf-8", errors="replace"
            )
            stderr = container.logs(stdout=False, stderr=True).decode(
                "utf-8", errors="replace"
            )

        return {
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": timed_out,
        }

    except (DockerException, Exception) as exc:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Docker error: {exc}",
            "timed_out": False,
        }
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except Exception:
                pass
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
