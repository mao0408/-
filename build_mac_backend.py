from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"


def main() -> None:
    DIST.mkdir(exist_ok=True)
    target = DIST / "backend"
    if target.exists():
        target.unlink()
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        "backend",
        "--distpath",
        str(DIST),
        "--workpath",
        str(ROOT / "build" / "pyinstaller-mac"),
        "--specpath",
        str(ROOT / "build"),
        "--collect-all",
        "rapidocr_onnxruntime",
        "--collect-all",
        "onnxruntime",
        "--collect-all",
        "cv2",
        "--collect-all",
        "PIL",
        "--hidden-import",
        "pyperclip",
        str(ROOT / "mac_backend.py"),
    ]
    subprocess.check_call(command, cwd=ROOT)
    if sys.platform == "darwin":
        target.chmod(0o755)
    print(f"Built backend: {target}")


if __name__ == "__main__":
    main()
