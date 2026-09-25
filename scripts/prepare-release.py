#!/usr/bin/env python3
"""Update CHANGELOG.md and CMakeLists.txt for a new release.

Usage: prepare-release.py <VERSION>

Called by semantic-release during the 'prepare' lifecycle step.
"""
import re
import subprocess
import sys
from datetime import date


def get_repo() -> str:
    """Derive 'owner/repo' from the git remote URL."""
    url = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], text=True
    ).strip()
    m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot parse GitHub repo from remote URL: {url}")


def update_cmake(version: str) -> None:
    """Replace the version inside the project() block in CMakeLists.txt."""
    with open("CMakeLists.txt", encoding="utf-8") as f:
        content = f.read()

    updated, n = re.subn(
        r"(project\(.*?VERSION\s+)\d+\.\d+\.\d+",
        rf"\g<1>{version}",
        content,
        count=1,
        flags=re.DOTALL,
    )
    if n == 0:
        print(
            "WARNING: VERSION not updated in CMakeLists.txt — pattern did not match",
            file=sys.stderr,
        )

    with open("CMakeLists.txt", "w", encoding="utf-8") as f:
        f.write(updated)


def update_changelog(version: str, release_date: str, repo: str) -> None:
    """Move content from ## [Unreleased] into a versioned section and refresh links."""
    with open("CHANGELOG.md", encoding="utf-8") as f:
        content = f.read()

    # Insert a versioned heading immediately after ## [Unreleased].
    content = content.replace(
        "## [Unreleased]",
        f"## [Unreleased]\n\n## [{version}] - {release_date}",
        1,
    )

    # Discover the previous latest version from the [Unreleased] compare link.
    prev_match = re.search(
        r"\[Unreleased\]: https://github\.com/.+/compare/v(.+?)\.\.\.HEAD",
        content,
    )
    prev_version = prev_match.group(1) if prev_match else None

    # Update the [Unreleased] compare link to start from the new tag.
    content, n = re.subn(
        r"\[Unreleased\]: .+",
        f"[Unreleased]: https://github.com/{repo}/compare/v{version}...HEAD",
        content,
    )

    if n > 0 and prev_version:
        # Insert the new version link directly after the [Unreleased] line.
        content = re.sub(
            r"(\[Unreleased\]: .+\n)",
            (
                r"\g<1>"
                f"[{version}]: https://github.com/{repo}/compare/v{prev_version}...v{version}\n"
            ),
            content,
        )
    elif n == 0:
        # No existing [Unreleased] link — append both new entries.
        content = (
            content.rstrip()
            + f"\n\n[Unreleased]: https://github.com/{repo}/compare/v{version}...HEAD\n"
            f"[{version}]: https://github.com/{repo}/releases/tag/v{version}\n"
        )

    with open("CHANGELOG.md", "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: prepare-release.py <VERSION>", file=sys.stderr)
        sys.exit(1)

    version = sys.argv[1]
    release_date = date.today().isoformat()
    repo = get_repo()

    update_cmake(version)
    update_changelog(version, release_date, repo)
    print(f"Prepared release {version}")


if __name__ == "__main__":
    main()
