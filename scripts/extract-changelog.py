#!/usr/bin/env python3
"""Print the body of the ## [Unreleased] section from CHANGELOG.md.

Output is used by semantic-release as the GitHub Release notes.
Exits silently with no output if the section is empty.
"""
import re
import sys

with open("CHANGELOG.md", encoding="utf-8") as f:
    content = f.read()

m = re.search(
    r"^## \[Unreleased\]\s*\n(.*?)(?=^## \[|\Z)",
    content,
    re.MULTILINE | re.DOTALL,
)
if not m:
    sys.exit(0)

notes = m.group(1).strip()
if notes:
    print(notes)
