"""Build a private CUDA 12.8/glibc overlay; never modify installed toolkit headers.

See https://forums.developer.nvidia.com/t/323591 . Narrow six-declaration
noexcept fix is qualified locally, not a general supported-toolchain assertion.
"""

import argparse
import hashlib
from pathlib import Path
import re
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    parser.error("output must be a fresh private directory")
shutil.copytree(args.source, args.output)
header = args.output / "crt/math_functions.h"
content = header.read_text()
original = hashlib.sha256(header.read_bytes()).hexdigest()
for name in ("rsqrt", "rsqrtf", "sinpi", "sinpif", "cospi", "cospif"):
    content, count = re.subn(r"(extern [^;\n]+\b"+name+r"\([^;\n]+\));", r"\1 noexcept (true);", content)
    if count != 1:
        raise ValueError(f"unexpected header; expected one declaration for {name}, got {count}")
header.write_text(content)
print({"original_sha256": original, "overlay_sha256": hashlib.sha256(header.read_bytes()).hexdigest()})
