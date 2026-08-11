import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
VENDORED_PACKAGES = {
    "mujoco_viewer": ROOT_DIR / "third_party/mujoco_viewer",
    "unitree_cpp": ROOT_DIR / "third_party/unitree_cpp",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Install vendored third-party Python packages")
    parser.add_argument(
        "packages",
        nargs="*",
        choices=sorted(VENDORED_PACKAGES),
        help="Packages to install; defaults to mujoco_viewer",
    )
    return parser.parse_args()


def main():
    selected = parse_args().packages or ["mujoco_viewer"]
    for name in selected:
        package_dir = VENDORED_PACKAGES[name]
        if not package_dir.is_dir():
            raise FileNotFoundError(f"Vendored package directory is missing: {package_dir}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", str(package_dir)],
            check=True,
        )


if __name__ == "__main__":
    main()
