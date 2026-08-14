import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
VENDORED_PACKAGES = {
    "unitree_cpp": ROOT_DIR / "third_party/unitree_cpp",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Install vendored third-party Python packages")
    parser.add_argument(
        "packages",
        nargs="*",
        choices=sorted(VENDORED_PACKAGES),
        help="Packages to install; defaults to unitree_cpp",
    )
    return parser.parse_args()


def main():
    selected = parse_args().packages or ["unitree_cpp"]
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
