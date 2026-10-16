#!/usr/bin/env python3
"""Print the body of a CHANGELOG.md section.

Usage: extract-changelog.py [VERSION]

With VERSION, prints the body of the ## [VERSION] section; without it, the
body of the ## [Unreleased] section. The release workflow uses the output as
the GitHub Release notes. Exits silently with no output if the section is
missing or empty.
"""
import re
import sys

heading = re.escape(sys.argv[1]) if len(sys.argv) > 1 else "Unreleased"

with open("CHANGELOG.md", encoding="utf-8") as f:
    content = f.read()

m = re.search(
    rf"^## \[{heading}\][^\n]*\n(.*?)(?=^## \[|^\[[^\]\n]+\]: |\Z)",
    content,
    re.MULTILINE | re.DOTALL,
)
if not m:
    sys.exit(0)

notes = m.group(1).strip()
if notes:
    print(notes)
