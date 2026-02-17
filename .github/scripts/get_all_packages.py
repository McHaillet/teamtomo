import json
import subprocess
from pathlib import Path


def get_all_packages() -> list[str]:
    """Get all workspace package names."""
    workspace_packages = []

    # Define workspace member glob patterns (same as pyproject.toml)
    patterns = [
        "packages/io/*/pyproject.toml",
        "packages/primitives/*/pyproject.toml",
        "packages/algorithms/*/pyproject.toml",
        "packages/wip/*/pyproject.toml",
    ]

    for pattern in patterns:
        for pyproject in Path(".").glob(pattern):
            # Parse pyproject.toml to get package name
            content = pyproject.read_text()

            for line in content.split("\n"):
                if line.startswith('name = "'):
                    pkg_name = line.split('"')[1]
                    workspace_packages.append(pkg_name)
                    break

    return sorted(workspace_packages)


if __name__ == "__main__":
    packages = get_all_packages()
    for pkg in packages:
        print(pkg)
