"""Build a source-only ZIP from Git-tracked files, with content checksums."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path


def build(output):
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=root,
                             capture_output=True, check=True).stdout.decode().split("\0")
    paths = [Path(name) for name in tracked if name]
    if not paths:
        raise ValueError("No tracked source files. Stage the reviewed source release first.")
    forbidden_dirs = {".git", ".venv", "data", "weights", "runs", "outputs", "__pycache__"}
    forbidden_extensions = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".docx", ".svs", ".ndpi", ".pem", ".key"}
    for path in paths:
        if forbidden_dirs.intersection(path.parts) or path.suffix.lower() in forbidden_extensions or path.name.startswith(".env"):
            raise ValueError(f"Non-submission file was tracked: {path}")
        if (root / path).is_symlink():
            raise ValueError(f"Symlinks are not included in the submission archive: {path}")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = "tls-multiclass-segmentation-quantification/"
    checksums = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            data = (root / path).read_bytes()
            name = path.as_posix()
            checksums.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
            info = zipfile.ZipInfo(prefix + name, date_time=(2026, 11, 25, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
        archive.writestr(prefix + "MANIFEST.sha256", "\n".join(checksums) + "\n")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(f"Built {output.name}: {len(paths)} source files; SHA256 {digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    build(parser.parse_args().output)
