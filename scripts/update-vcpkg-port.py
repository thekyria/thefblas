#!/usr/bin/env python3
"""Point the vcpkg port and versions database at a published release.

Usage: update-vcpkg-port.py <VERSION>

Run from the repository root after the v<VERSION> tag exists on GitHub. The
script downloads the tag's tarball, writes its SHA512 into
ports/thefblas/portfile.cmake, sets the version in ports/thefblas/vcpkg.json,
and records the port's git tree in versions/t-/thefblas.json and
versions/baseline.json. It stages ports/thefblas/ to compute that tree.

The port cannot be updated in the tagged commit itself, because the tarball's
hash depends on that commit's contents; commit the result in a follow-up PR.
"""
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request

PORTFILE = "ports/thefblas/portfile.cmake"
PORT_MANIFEST = "ports/thefblas/vcpkg.json"
VERSIONS = "versions/t-/thefblas.json"
BASELINE = "versions/baseline.json"


def get_repo() -> str:
    """Derive 'owner/repo' from the git remote URL."""
    url = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], text=True
    ).strip()
    m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot parse GitHub repo from remote URL: {url}")


def tarball_sha512(url: str, attempts: int = 5) -> str:
    """SHA512 of the file at url, retrying transient failures."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return hashlib.sha512(response.read()).hexdigest()
        except OSError as error:
            if attempt == attempts:
                raise
            print(f"Download failed ({error}); retrying", file=sys.stderr)
            time.sleep(5)
    raise AssertionError("unreachable")


def substitute(path: str, pattern: str, replacement: str) -> None:
    """Replace the single match of pattern in path, or fail."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    updated, n = re.subn(pattern, replacement, content, count=1)
    if n == 0:
        raise SystemExit(f"error: pattern {pattern!r} not found in {path}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)


def update_json(path: str, update) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    update(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: update-vcpkg-port.py <VERSION>", file=sys.stderr)
        sys.exit(1)
    version = sys.argv[1]
    url = f"https://github.com/{get_repo()}/archive/refs/tags/v{version}.tar.gz"

    sha512 = tarball_sha512(url)
    substitute(PORTFILE, r"SHA512\s+[0-9a-f]+", f"SHA512\n  {sha512}")
    substitute(PORT_MANIFEST, r'"version":\s*"[^"]*"', f'"version": "{version}"')

    subprocess.check_call(["git", "add", "ports/thefblas/"])
    tree = subprocess.check_output(
        ["git", "write-tree", "--prefix=ports/thefblas/"], text=True
    ).strip()

    def add_version(data):
        entries = [e for e in data.get("versions", []) if e.get("version") != version]
        entries.insert(0, {"version": version, "git-tree": tree})
        data["versions"] = entries

    def set_baseline(data):
        data["default"]["thefblas"] = {"baseline": version, "port-version": 0}

    update_json(VERSIONS, add_version)
    update_json(BASELINE, set_baseline)
    print(f"vcpkg port updated to {version} (SHA512 {sha512[:16]}..., git-tree {tree})")


if __name__ == "__main__":
    main()
