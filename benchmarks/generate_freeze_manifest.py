"""
==============================================================================
TRACEBIND v1.0 Cryptographic Freeze Generator
==============================================================================
Computes SHA-256 checksums for script and result directory artifacts to populate
BENCHMARK_FREEZE_v1.md.
==============================================================================
"""

import os
import hashlib

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(BENCHMARK_DIR, "run_phase3_benchmarks.py")
RESULTS_DIR = os.path.join(BENCHMARK_DIR, "phase3_results")
MANIFEST_PATH = os.path.join(BENCHMARK_DIR, "BENCHMARK_FREEZE_v1.md")

def get_file_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()

def get_dir_sha256(dirpath):
    sha = hashlib.sha256()
    for root, _, files in sorted(os.walk(dirpath)):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            sha.update(fname.encode("utf-8"))
            with open(fpath, "rb") as f:
                while chunk := f.read(8192):
                    sha.update(chunk)
    return sha.hexdigest()

script_hash = get_file_sha256(SCRIPT_PATH)
results_hash = get_dir_sha256(RESULTS_DIR)

print("=" * 60)
print(f"TRACEBIND Benchmark v1.0 Cryptographic Hashes:")
print(f"  run_phase3_benchmarks.py SHA-256: {script_hash}")
print(f"  phase3_results/ SHA-256:          {results_hash}")
print("=" * 60)

if os.path.exists(MANIFEST_PATH):
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    content = content.replace("[INSERT_SCRIPT_HASH]", script_hash)
    content = content.replace("[INSERT_RESULTS_HASH]", results_hash)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print("✓ BENCHMARK_FREEZE_v1.md successfully updated with cryptographic hashes!")